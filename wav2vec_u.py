# =============================================================================
# wav2vec_u.py  — Wav2Vec-U GAN Model (Refactored for Separation of Concerns)
# =============================================================================
#
# FILE LOCATION (in your forked repo):
#   wav2vec_unsupervised/fairseq_/examples/wav2vec/unsupervised/models/wav2vec_u.py
#
# WHAT THIS FILE DOES (plain English):
#   Wav2Vec-U trains a GAN (Generative Adversarial Network) to do speech
#   recognition WITHOUT labelled data (no audio-text pairs needed).
#
#   How the GAN works here:
#     1. GENERATOR  — takes raw speech features (from wav2vec 2.0 encoder)
#                     and produces a sequence of "fake" phoneme probabilities.
#                     Think of it as trying to convert sound into text-like units.
#
#     2. REALDATA   — reads real, unlabelled text (e.g. LibriSpeech sentences)
#                     and provides "real" phoneme sequences as reference.
#                     The model never sees which audio matches which text!
#
#     3. DISCRIMINATOR — a judge that looks at a sequence and decides:
#                        "Is this real text or fake speech-converted-to-phonemes?"
#                        As training progresses, the Generator gets better at
#                        fooling the Discriminator → it learns to transcribe!
#
# ARCHITECTURE DIAGRAM:
#
#   [Audio] → wav2vec2 encoder → [Speech Features 512-dim]
#                                         │
#                                    GENERATOR
#                                    (Conv1d layers)
#                                         │
#                              [Fake phoneme sequence]
#                                         │
#                                  DISCRIMINATOR ←── [Real phoneme text]
#                                         │                  │
#                                    (judge)            REALDATA
#                                         │
#                               Loss: fool the judge
#
# DEVICE SELECTION (CPU / MPS / CUDA):
#   On Mac M1/M2/M3 you have three choices:
#     • "cpu"  — always works, slowest
#     • "mps"  — Apple Silicon GPU (fast, set DEVICE=mps in config)
#     • "cuda" — NVIDIA GPU (Linux/Windows only)
#   Change DEVICE at the bottom of this file or pass it via the training config.
#
# DATASET CONFIGURATION:
#   By default this uses LibriSpeech-mini (tiny 1h subset, easy to download).
#   To switch datasets, change the DATASET_CONFIG dict at the bottom.
#   Supported out-of-the-box:
#     • "librispeech_mini"  — ~1 GB, best for development
#     • "librispeech_100"   — 100h, better results
#     • "custom"            — point to your own .wav folder + text file
#
# DEPENDENCIES (install once):
#   pip install torch torchaudio datasets tqdm
#   pip install fairseq   # or use the repo's local fairseq_ copy
#
# =============================================================================

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # progress bars

# ── Fairseq imports (these come from the repo's local fairseq_ folder) ────────
from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 0 — DEVICE SELECTION
# =============================================================================
# Easily switch between CPU, Apple Silicon GPU (MPS), and NVIDIA GPU (CUDA).
# The helper below is called once at model creation time.

def select_device(preferred: str = "auto") -> torch.device:
    """
    Choose the best available compute device.

    Args:
        preferred: "auto" | "cpu" | "mps" | "cuda"
                   "auto" = picks CUDA > MPS > CPU in that order.

    Returns:
        torch.device ready to use.

    Mac users:
        Pass preferred="mps" to use your Apple Silicon GPU.
        If MPS is not available, falls back to CPU with a warning.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif preferred == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS (Apple Silicon GPU) not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    logger.info(f"[Device] Using: {device}")
    return device


# =============================================================================
# SECTION 1 — DATASET CONFIGURATION
# =============================================================================
# Change DATASET_NAME to switch between datasets.
# Everything downstream (data loading, phoneme vocab size) adapts automatically.

DATASET_CONFIG: Dict[str, Any] = {
    # ── Active dataset ────────────────────────────────────────────────────────
    # Change this string to switch datasets:
    #   "librispeech_mini" | "librispeech_100" | "custom"
    "active": "librispeech_mini",

    # ── LibriSpeech Mini (recommended for development / this assignment) ──────
    "librispeech_mini": {
        "description": "Tiny LibriSpeech subset (~1h), great for quick experiments",
        "hf_dataset":   "hf-internal-testing/librispeech_asr_dummy",  # HuggingFace dataset
        "hf_split":     "validation",
        "text_column":  "text",
        "audio_column": "audio",
        "phoneme_vocab_size": 41,   # standard English phoneme set (ARPAbet)
        "sample_rate":  16_000,
    },

    # ── LibriSpeech 100h (better quality, larger download) ───────────────────
    "librispeech_100": {
        "description": "LibriSpeech train-clean-100 (100 hours of English)",
        "hf_dataset":  "librispeech_asr",
        "hf_split":    "train.clean.100",
        "text_column": "text",
        "audio_column": "audio",
        "phoneme_vocab_size": 41,
        "sample_rate": 16_000,
    },

    # ── Custom dataset ────────────────────────────────────────────────────────
    # Set "audio_dir" to a folder of .wav files and
    # "text_file" to a plain text file (one sentence per line).
    "custom": {
        "description": "Your own audio + text (unpaired)",
        "audio_dir":   "/path/to/your/wav/files",
        "text_file":   "/path/to/your/text.txt",
        "phoneme_vocab_size": 41,
        "sample_rate": 16_000,
    },
}


def get_active_dataset_cfg() -> Dict[str, Any]:
    """Return the configuration dict for whichever dataset is active."""
    active = DATASET_CONFIG["active"]
    cfg = DATASET_CONFIG.get(active)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset '{active}'. "
            f"Valid options: {list(DATASET_CONFIG.keys())[1:]}"
        )
    logger.info(f"[Dataset] Active: {active} — {cfg.get('description','')}")
    return cfg


# =============================================================================
# SECTION 2 — MODEL CONFIGURATION (hyperparameters)
# =============================================================================


@dataclass
class SegmentationConfig(FairseqDataclass):
    """Matches `model.segmentation` in config/gan/w2vu.yaml."""

    type: str = field(default="JOIN", metadata={"help": "Segmentation strategy."})
    mean_pool_join: bool = False
    remove_zeros: bool = False


@dataclass
class Wav2VecUConfig(FairseqDataclass):
    """
    All the knobs you can turn for the Wav2Vec-U GAN model.
    These are read from the YAML config file (config/gan/w2vu.yaml).

    You generally do NOT need to edit this class directly —
    change the YAML file instead.
    """

    # ── Generator hyperparameters ─────────────────────────────────────────────
    generator_embed_dim: int = field(
        default=128,
        metadata={"help": "Size of the Generator's internal hidden state. "
                          "Larger = more capacity but slower."}
    )
    generator_hidden: int = field(
        default=512,
        metadata={"help": "Number of channels in Generator convolution layers."}
    )
    generator_kernel: int = field(
        default=4,
        metadata={"help": "Convolution kernel size in Generator. "
                          "Controls how many time-steps the Generator sees at once."}
    )
    generator_dilation: int = field(
        default=1,
        metadata={"help": "Dilation of Generator convolutions. "
                          ">1 increases receptive field without more parameters."}
    )
    generator_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability in Generator (0 = off)."}
    )
    generator_batch_norm: int = field(
        default=0,
        metadata={"help": "Apply batch normalisation every N layers in Generator. "
                          "0 = off."}
    )
    generator_residual: bool = field(
        default=False,
        metadata={"help": "Add residual (skip) connections in Generator."}
    )
    generator_stride: int = field(
        default=1,
        metadata={"help": "Conv1d stride in Generator (w2vu.yaml / upstream)."},
    )
    generator_bias: bool = field(
        default=False,
        metadata={"help": "Whether Generator Conv1d layers use bias (w2vu.yaml)."},
    )

    # ── Discriminator hyperparameters ─────────────────────────────────────────
    discriminator_type: str = field(
        default="conv",
        metadata={"help": "'conv' (1D CNN) or 'transformer'. "
                          "CNN is faster; Transformer is more powerful."}
    )
    discriminator_dim: int = field(
        default=256,
        metadata={"help": "Hidden size of Discriminator."}
    )
    discriminator_kernel: int = field(
        default=8,
        metadata={"help": "Convolution kernel size in Discriminator."}
    )
    discriminator_dilation: int = field(
        default=1,
        metadata={"help": "Dilation in Discriminator convolutions."}
    )
    discriminator_depth: int = field(
        default=3,
        metadata={"help": "Number of layers in Discriminator."}
    )
    discriminator_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout in Discriminator (0 = off)."}
    )
    discriminator_spectral_norm: bool = field(
        default=False,
        metadata={"help": "Apply spectral normalisation to Discriminator weights. "
                          "Helps training stability."}
    )
    discriminator_weight_norm: bool = field(
        default=False,
        metadata={"help": "Apply weight normalisation (alternative to spectral norm)."}
    )
    discriminator_linear_emb: bool = field(
        default=False,
        metadata={"help": "Use a linear projection as embedding instead of learned."}
    )
    discriminator_causal: bool = field(
        default=True,
        metadata={"help": "Use causal (one-directional) convolutions in Discriminator. "
                          "True = each step can only see past steps."}
    )
    discriminator_max_pool: bool = field(
        default=False,
        metadata={"help": "Use max pooling instead of mean pooling for final readout."}
    )
    discriminator_act_after_linear: bool = field(
        default=False,
        metadata={"help": "Apply activation after linear projection in Discriminator."}
    )

    # ── Training loss weights ─────────────────────────────────────────────────
    smoothness_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for the smoothness penalty loss. "
                          "Penalises rapid jumps in Generator output. "
                          "Encourages more natural-sounding sequences."}
    )
    smoothing_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for label-smoothing loss on Generator."}
    )
    smoothing: float = field(
        default=0.0,
        metadata={"help": "Alternate smoothing knob from w2vu.yaml (upstream name)."},
    )
    smoothing_one_sided: bool = field(
        default=False,
        metadata={"help": "One-sided label smoothing (w2vu.yaml)."},
    )
    gumbel: bool = field(
        default=False,
        metadata={"help": "Gumbel-softmax (w2vu.yaml)."},
    )
    hard_gumbel: bool = field(
        default=False,
        metadata={"help": "Hard Gumbel (w2vu.yaml)."},
    )
    temp: List[float] = field(
        default_factory=lambda: [2.0, 0.1, 0.99995],
        metadata={"help": "Temperature schedule [start, end, decay] (w2vu.yaml)."},
    )
    input_dim: int = field(
        default=512,
        metadata={"help": "Wav2vec feature dimension fed to the Generator."},
    )
    mmi_weight: float = field(
        default=0.0,
        metadata={"help": "MMI-style auxiliary loss weight (w2vu2.yaml)."},
    )
    target_dim: int = field(
        default=64,
        metadata={"help": "Auxiliary target dimension (w2vu2.yaml)."},
    )
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    code_penalty: float = field(
        default=0.0,
        metadata={"help": "Penalty to encourage diversity in Generator outputs."}
    )

    # ── Gradient penalty (Wasserstein GAN style) ──────────────────────────────
    gradient_penalty: float = field(
        default=0.0,
        metadata={"help": "Gradient penalty coefficient for Discriminator training. "
                          ">0 enables WGAN-GP style training."}
    )

    # ── Probabilistic smoothing ───────────────────────────────────────────────
    probabilistic_grad_penalty_sched: bool = field(
        default=False,
        metadata={"help": "Randomly skip gradient penalty computation by schedule."}
    )

    # ── Segment length for GAN training ──────────────────────────────────────
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": "Crop sequences to this many frames before feeding GAN. "
                          "Helps with memory on long utterances."}
    )
    min_length: int = field(
        default=0,
        metadata={"help": "Skip utterances shorter than this (frames)."}
    )

    # ── Discriminator update schedule ─────────────────────────────────────────
    discriminator_updates: int = field(
        default=1,
        metadata={"help": "How many Discriminator update steps per Generator step."}
    )

    # ── Segment-level features ─────────────────────────────────────────────────
    segmented_output_layer: int = field(
        default=-1,
        metadata={"help": "Which layer of wav2vec 2.0 to use as input features. "
                          "-1 = last layer."}
    )

    # ── Device preference ─────────────────────────────────────────────────────
    device: str = field(
        default="auto",
        metadata={"help": "Compute device: 'auto' | 'cpu' | 'mps' | 'cuda'. "
                          "Mac users: set to 'mps' for Apple Silicon GPU."}
    )


# =============================================================================
# SECTION 3 — GENERATOR CLASS
# =============================================================================
#
# PURPOSE: Convert raw speech features → fake phoneme probability sequences.
#
# HOW IT WORKS:
#   The Generator receives a tensor of shape [Batch, Time, Features]:
#     • Batch   = number of audio clips in this mini-batch
#     • Time    = number of frames (e.g. 100 frames ≈ 1 second)
#     • Features = 512-dimensional wav2vec 2.0 embeddings
#
#   It passes these through a stack of 1D convolutions.
#   Each convolution scans along the Time axis, learning local patterns.
#   The final layer outputs a probability distribution over phonemes
#   (e.g. 41 English phonemes + silence + blank).
#
# ANALOGY:
#   Think of the Generator as a student who listens to audio and writes
#   down their best guess at what phonemes were spoken, one frame at a time.

class Generator(nn.Module):
    """
    Generator: Speech features → Fake phoneme probability sequence.

    This is one of the three core classes required by the assignment.
    It represents the "student" in the GAN game — trying to fool the
    Discriminator into thinking its fake phoneme sequences are real text.

    Args:
        input_dim:   Dimensionality of input features (wav2vec 2.0 output, usually 512).
        output_dim:  Number of phoneme classes (vocabulary size, e.g. 41).
        cfg:         Wav2VecUConfig with all hyperparameters.
        normalize_fn: Optional function to normalise inputs before processing.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: Wav2VecUConfig,
        normalize_fn=None,
    ):
        super().__init__()

        self.input_dim    = input_dim
        self.output_dim   = output_dim
        self.cfg          = cfg
        self.normalize_fn = normalize_fn  # e.g. instance norm

        # ── Build the convolutional stack ─────────────────────────────────────
        #    Each layer: Conv1d → optional BatchNorm → ELU activation
        #    The last layer maps to phoneme vocabulary size.

        inner_dim = cfg.generator_hidden  # e.g. 512

        # Project input features to inner_dim
        self.input_proj = nn.Linear(input_dim, inner_dim)

        # Build the main conv stack
        self.convolutions = self._build_conv_stack(
            in_channels  = inner_dim,
            hidden_channels = inner_dim,
            out_channels = cfg.generator_embed_dim,
            kernel_size  = cfg.generator_kernel,
            dilation     = cfg.generator_dilation,
            stride       = cfg.generator_stride,
            bias         = cfg.generator_bias,
            num_layers   = 4,  # four conv layers, as in original Wav2Vec-U
            dropout      = cfg.generator_dropout,
            use_bn       = cfg.generator_batch_norm > 0,
            use_residual = cfg.generator_residual,
        )

        # Final projection to phoneme vocabulary
        self.output_proj = nn.Linear(cfg.generator_embed_dim, output_dim)

        # Dropout after each conv (regularisation to prevent overfitting)
        self.dropout = nn.Dropout(p=cfg.generator_dropout)

        logger.info(
            f"[Generator] Built: input_dim={input_dim}, "
            f"hidden={inner_dim}, output_dim={output_dim}, "
            f"kernel={cfg.generator_kernel}, layers=4"
        )

    def _build_conv_stack(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        stride: int,
        bias: bool,
        num_layers: int,
        dropout: float,
        use_bn: bool,
        use_residual: bool,
    ) -> nn.ModuleList:
        """
        Create a list of 1D convolution blocks.

        Each block is:  Conv1d → (optional) BatchNorm → ELU

        Why Conv1d?
            Audio is a 1D signal over time. Conv1d slides a filter
            along the time axis, learning local patterns (e.g. phoneme
            transitions) without needing to look at the entire sequence.
        """
        layers = nn.ModuleList()
        current_in = in_channels

        for i in range(num_layers):
            # Last layer reduces to out_channels
            current_out = out_channels if i == num_layers - 1 else hidden_channels

            block = nn.ModuleDict({
                # padding = (kernel-1)*dilation keeps the output length the same as input
                "conv": nn.Conv1d(
                    in_channels  = current_in,
                    out_channels = current_out,
                    kernel_size  = kernel_size,
                    dilation     = dilation,
                    stride       = stride,
                    bias         = bias,
                    padding      = (kernel_size - 1) * dilation // 2,
                ),
                "act": nn.ELU(),  # ELU: smooth version of ReLU, works well for audio
            })

            if use_bn:
                block["bn"] = nn.BatchNorm1d(current_out)

            layers.append(block)
            current_in = current_out

        return layers

    def forward(
        self,
        features: torch.Tensor,          # [Batch, Time, Features]
        padding_mask: Optional[torch.Tensor] = None,  # True where padded
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert speech features into fake phoneme probabilities.

        Args:
            features:     [B, T, C] — wav2vec 2.0 output embeddings
            padding_mask: [B, T]    — True at padded positions (ignore these)

        Returns:
            x:      [B, T, V] — log-probability over V phonemes at each time step
            dense:  [B, T, V] — same, before padding is zeroed (needed for loss)
            padding_mask: aligned to output time steps (may differ from input after convs)

        Step-by-step:
            1. Optionally normalise features (reduces covariate shift)
            2. Project features from input_dim → inner_dim
            3. Transpose to [B, C, T] for Conv1d (Conv1d expects channels-first)
            4. Pass through conv stack
            5. Transpose back to [B, T, C]
            6. Project to phoneme vocabulary
            7. Apply log_softmax to get log-probabilities
        """
        # Step 1: Normalise
        if self.normalize_fn is not None:
            features, _ = self.normalize_fn(features)

        # Step 2: Input projection [B, T, C] → [B, T, inner_dim]
        x = self.input_proj(features)
        x = F.elu(x)

        # Step 3: Transpose for Conv1d → [B, inner_dim, T]
        x = x.transpose(1, 2)

        # Step 4: Pass through conv stack
        for block in self.convolutions:
            residual = x  # save for skip connection

            x = block["conv"](x)          # convolution
            if "bn" in block:
                x = block["bn"](x)        # batch norm (optional)
            x = block["act"](x)           # activation
            x = self.dropout(x)

            # Skip connection: add input to output if shapes match
            if self.cfg.generator_residual and residual.shape == x.shape:
                x = x + residual

        # Step 5: Transpose back → [B, T, inner_dim]
        x = x.transpose(1, 2)

        # Step 6: Project to vocabulary size → [B, T, vocab_size]
        x = self.output_proj(x)

        # Conv1d stack can change sequence length vs. input features; align mask.
        pm = padding_mask
        if pm is not None:
            T = x.size(1)
            if pm.size(1) != T:
                if pm.size(1) > T:
                    pm = pm[:, :T]
                else:
                    pm = torch.cat(
                        [
                            pm,
                            pm.new_ones(pm.size(0), T - pm.size(1)),
                        ],
                        dim=1,
                    )

        # Step 7: Log-softmax over phoneme dimension
        #   log_softmax(x)[i] = log(P(phoneme_i | frame))
        dense = x  # keep un-masked for loss computation
        x = F.log_softmax(x, dim=-1)

        # Zero out padded positions so they don't contribute to loss
        if pm is not None:
            x = x * (~pm).unsqueeze(-1).float()

        return x, dense, pm


# =============================================================================
# SECTION 4 — REALDATA CLASS
# =============================================================================
#
# PURPOSE: Provide real phoneme sequences from unlabelled text.
#
# HOW IT WORKS:
#   RealData reads from a corpus of plain text (no audio needed!).
#   It converts words → phonemes using a phoneme dictionary.
#   These phoneme sequences are what the Discriminator considers "real".
#
# ANALOGY:
#   RealData is like the teacher's answer key — it shows what correct
#   phoneme sequences look like, but crucially, this key was made from
#   text ONLY, not from the audio. So the GAN never sees paired data!
#
# DATASET SWITCHING:
#   The dataset used here is controlled by DATASET_CONFIG at the top.
#   Change DATASET_CONFIG["active"] to switch datasets.

class RealData(nn.Module):
    """
    RealData: Provides real phoneme distributions from unlabelled text.

    This class loads real text data and represents it as embeddings that
    the Discriminator uses as its "positive" examples.

    In the original Wav2Vec-U, real data is loaded as pre-processed phoneme
    sequences from the fairseq data pipeline. Here we make it self-contained
    so you can run it without the full fairseq infrastructure.

    Args:
        num_phonemes: Size of the phoneme vocabulary (e.g. 41 for English).
        embed_dim:    Embedding dimension for phonemes.
        dataset_cfg:  Dataset configuration dict (from DATASET_CONFIG).
    """

    def __init__(
        self,
        num_phonemes: int,
        embed_dim: int,
        dataset_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.num_phonemes = num_phonemes
        self.embed_dim    = embed_dim
        self.dataset_cfg  = dataset_cfg or get_active_dataset_cfg()

        # ── Phoneme Embedding Table ───────────────────────────────────────────
        #    Maps each phoneme index → a dense vector of size embed_dim.
        #    These are learned during training.
        self.phoneme_embed = nn.Embedding(
            num_embeddings = num_phonemes,
            embedding_dim  = embed_dim,
            padding_idx    = 0,   # index 0 = padding / silence
        )

        # ── Positional bias (helps the model understand sequence order) ───────
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))

        logger.info(
            f"[RealData] Built: vocab={num_phonemes}, "
            f"embed_dim={embed_dim}, "
            f"dataset={self.dataset_cfg.get('description', 'unknown')}"
        )

    def forward(
        self,
        phoneme_ids: torch.Tensor,        # [Batch, SeqLen] — integer phoneme indices
        padding_mask: Optional[torch.Tensor] = None,  # True where padded
    ) -> torch.Tensor:
        """
        Convert phoneme index sequences into dense embeddings.

        Args:
            phoneme_ids:  [B, T] — integer indices from phoneme vocabulary
            padding_mask: [B, T] — True at padded positions

        Returns:
            embeddings: [B, T, embed_dim] — dense real phoneme representations

        These embeddings are passed to the Discriminator as "real" samples.
        The Discriminator must learn to recognise this pattern and reject
        the Generator's fakes.
        """
        # Look up phoneme embeddings
        embeddings = self.phoneme_embed(phoneme_ids)     # [B, T, embed_dim]

        # Add positional bias (helps model distinguish positions)
        embeddings = embeddings + self.pos_bias

        # Zero out padded positions
        if padding_mask is not None:
            embeddings = embeddings * (~padding_mask).unsqueeze(-1).float()

        return embeddings

    @staticmethod
    def load_text_dataset(dataset_cfg: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Load raw text sentences from the configured dataset.

        This method is called during data preparation (not during the forward pass).
        It returns a list of raw text strings that will be converted to phonemes
        by the data pipeline.

        Returns:
            List of text strings, e.g. ["hello world", "the cat sat", ...]

        Usage example:
            sentences = RealData.load_text_dataset()
            print(f"Loaded {len(sentences)} sentences")
        """
        cfg = dataset_cfg or get_active_dataset_cfg()

        sentences = []
        active = DATASET_CONFIG["active"]

        if active in ("librispeech_mini", "librispeech_100"):
            # Load from HuggingFace datasets
            # Note: requires `pip install datasets`
            try:
                from datasets import load_dataset
                logger.info(f"[RealData] Downloading {cfg['hf_dataset']} from HuggingFace...")

                # tqdm progress via datasets' built-in logging
                dataset = load_dataset(
                    cfg["hf_dataset"],
                    split=cfg["hf_split"],
                    trust_remote_code=True,
                )

                text_col = cfg["text_column"]
                sentences = [row[text_col].lower() for row in tqdm(
                    dataset,
                    desc="[RealData] Reading text",
                    unit=" sentences",
                )]

                logger.info(f"[RealData] Loaded {len(sentences):,} sentences.")

            except ImportError:
                logger.error(
                    "[RealData] 'datasets' library not found. "
                    "Install with: pip install datasets"
                )
                raise

        elif active == "custom":
            text_file = cfg.get("text_file", "")
            if not os.path.isfile(text_file):
                raise FileNotFoundError(
                    f"[RealData] Custom text file not found: {text_file}\n"
                    f"Set DATASET_CONFIG['custom']['text_file'] to your text path."
                )
            with open(text_file, "r", encoding="utf-8") as f:
                sentences = [
                    line.strip().lower()
                    for line in tqdm(f, desc="[RealData] Reading custom text")
                    if line.strip()
                ]
            logger.info(f"[RealData] Loaded {len(sentences):,} sentences from {text_file}.")

        else:
            raise ValueError(f"[RealData] Unknown dataset: {active}")

        return sentences


# =============================================================================
# SECTION 5 — DISCRIMINATOR CLASS
# =============================================================================
#
# PURPOSE: Judge whether a phoneme sequence is "real" (from text) or
#          "fake" (generated by the Generator from audio).
#
# HOW IT WORKS:
#   The Discriminator receives a sequence of vectors:
#     • "Real" vectors come from RealData (text-derived phoneme embeddings)
#     • "Fake" vectors come from Generator (speech-derived phoneme probs)
#
#   It passes them through a stack of 1D convolutions, then
#   produces a single scalar score: high = real, low = fake.
#
# TRAINING DYNAMIC:
#   • Discriminator wants: score(real) → high, score(fake) → low
#   • Generator wants:     score(fake) → high  (fool the Discriminator)
#   • This adversarial tension drives the Generator to produce
#     increasingly realistic phoneme sequences → ASR!
#
# ANALOGY:
#   The Discriminator is like an expert linguist who has read thousands of
#   transcripts and knows what natural phoneme patterns look like. The
#   Generator is a student trying to write convincing fake transcripts.

class Discriminator(nn.Module):
    """
    Discriminator: Judges whether a phoneme sequence is real or fake.

    This is one of the three core classes required by the assignment.
    It outputs a score per sequence (higher = more likely real).

    Two types are supported (set via cfg.discriminator_type):
        "conv"        — 1D CNN (fast, default)
        "transformer" — Transformer encoder (more powerful, slower)

    Args:
        input_dim: Dimensionality of the input sequences.
        cfg:       Wav2VecUConfig with all hyperparameters.
    """

    def __init__(self, input_dim: int, cfg: Wav2VecUConfig):
        super().__init__()

        self.cfg       = cfg
        self.input_dim = input_dim
        inner_dim      = cfg.discriminator_dim  # e.g. 256

        # ── Input projection ──────────────────────────────────────────────────
        #    Map whatever input_dim we receive to the Discriminator's inner dim.
        if cfg.discriminator_linear_emb:
            self.input_proj = nn.Linear(input_dim, inner_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, inner_dim),
                nn.ELU() if cfg.discriminator_act_after_linear else nn.Identity(),
            )

        # ── Main body ─────────────────────────────────────────────────────────
        if cfg.discriminator_type == "conv":
            self.body = self._build_conv_discriminator(inner_dim, cfg)
        elif cfg.discriminator_type == "transformer":
            self.body = self._build_transformer_discriminator(inner_dim, cfg)
        else:
            raise ValueError(
                f"Unknown discriminator_type: {cfg.discriminator_type}. "
                f"Use 'conv' or 'transformer'."
            )

        # ── Output head ───────────────────────────────────────────────────────
        #    Single linear layer → scalar score
        self.output_head = nn.Linear(inner_dim, 1)

        # ── Dropout ───────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=cfg.discriminator_dropout)

        logger.info(
            f"[Discriminator] Built: type={cfg.discriminator_type}, "
            f"input_dim={input_dim}, hidden={inner_dim}, "
            f"depth={cfg.discriminator_depth}, "
            f"kernel={cfg.discriminator_kernel}"
        )

    # ── Private builder methods ───────────────────────────────────────────────

    def _apply_norm(self, layer: nn.Module) -> nn.Module:
        """
        Optionally wrap a layer with spectral or weight normalisation.

        Spectral norm: constrains the largest singular value of the weight
                       matrix, preventing unstable training (mode collapse).
        Weight norm:   re-parameterises weights for faster convergence.
        """
        if self.cfg.discriminator_spectral_norm:
            return nn.utils.spectral_norm(layer)
        elif self.cfg.discriminator_weight_norm:
            return nn.utils.weight_norm(layer)
        return layer

    def _build_conv_discriminator(
        self, inner_dim: int, cfg: Wav2VecUConfig
    ) -> nn.Sequential:
        """
        Build a stack of 1D convolutional layers.

        Each layer: (optional norm) Conv1d → ELU → Dropout

        Causal vs non-causal convolution:
            Causal  (cfg.discriminator_causal=True):
                Padding only on the left → each output only sees PAST frames.
                Mimics left-to-right reading of text.
            Non-causal (False):
                Symmetric padding → each output sees past AND future.
        """
        layers = []
        depth  = cfg.discriminator_depth

        for i in range(depth):
            if cfg.discriminator_causal:
                # For causal: pad exactly (kernel-1)*dilation on the left only
                pad = (cfg.discriminator_kernel - 1) * cfg.discriminator_dilation
                conv = self._apply_norm(
                    nn.Conv1d(
                        in_channels  = inner_dim,
                        out_channels = inner_dim,
                        kernel_size  = cfg.discriminator_kernel,
                        dilation     = cfg.discriminator_dilation,
                        padding      = 0,    # we'll pad manually
                    )
                )
                layers.append(CausalConv1dBlock(conv, pad, nn.ELU()))
            else:
                # Non-causal: symmetric padding
                pad = (cfg.discriminator_kernel - 1) * cfg.discriminator_dilation // 2
                conv = self._apply_norm(
                    nn.Conv1d(
                        inner_dim, inner_dim,
                        kernel_size = cfg.discriminator_kernel,
                        dilation    = cfg.discriminator_dilation,
                        padding     = pad,
                    )
                )
                layers.append(nn.Sequential(conv, nn.ELU(), nn.Dropout(cfg.discriminator_dropout)))

        return nn.Sequential(*layers)

    def _build_transformer_discriminator(
        self, inner_dim: int, cfg: Wav2VecUConfig
    ) -> nn.TransformerEncoder:
        """
        Build a Transformer encoder as the Discriminator body.

        The Transformer uses self-attention — each time step can attend to
        ALL other time steps simultaneously, giving it a global view of the
        sequence. This makes it more powerful than CNNs but slower.

        Use this when you have a GPU and want maximum accuracy.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = inner_dim,
            nhead           = max(1, inner_dim // 64),   # number of attention heads
            dim_feedforward = inner_dim * 4,
            dropout         = cfg.discriminator_dropout,
            batch_first     = True,     # expects [B, T, C] input
        )
        return nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = cfg.discriminator_depth,
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, C]
        padding_mask: Optional[torch.Tensor] = None,  # [B, T] True=pad
    ) -> torch.Tensor:
        """
        Score a batch of phoneme sequences.

        Args:
            x:            [B, T, C] — either real (from RealData) or
                                      fake (from Generator) sequences
            padding_mask: [B, T]   — True at padded positions

        Returns:
            scores: [B] — one scalar score per sequence in the batch
                          Higher score → Discriminator thinks it's more "real"

        Step-by-step:
            1. Project input to inner_dim
            2. Pass through CNN or Transformer body
            3. Pool over the time dimension → single vector per sequence
            4. Linear head → scalar score
        """
        # Step 1: Project input
        x = self.input_proj(x)                        # [B, T, inner_dim]
        x = self.dropout(x)

        # Step 2: Pass through body (CNN or Transformer)
        if self.cfg.discriminator_type == "conv":
            # Conv1d expects [B, C, T]
            x = x.transpose(1, 2)                     # [B, inner_dim, T]
            x = self.body(x)                          # [B, inner_dim, T]
            x = x.transpose(1, 2)                     # [B, T, inner_dim]

        else:  # transformer
            # TransformerEncoder (batch_first=True) expects [B, T, C]
            src_key_padding_mask = padding_mask        # True = ignore
            x = self.body(x, src_key_padding_mask=src_key_padding_mask)  # [B, T, inner_dim]

        # Step 3: Pool over time axis → [B, inner_dim]
        if padding_mask is not None:
            # Don't pool over padded positions — they're meaningless
            mask_float = (~padding_mask).unsqueeze(-1).float()  # [B, T, 1]
            x = x * mask_float

        if self.cfg.discriminator_max_pool:
            # Max pooling: take the single most "real-looking" frame
            x = x.max(dim=1).values                   # [B, inner_dim]
        else:
            # Mean pooling: average across all frames
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1, keepdim=True).float()  # [B, 1]
                x = x.sum(dim=1) / lengths.clamp(min=1)
            else:
                x = x.mean(dim=1)                     # [B, inner_dim]

        # Step 4: Output head → [B, 1] → squeeze → [B]
        scores = self.output_head(x).squeeze(-1)      # [B]
        return scores


# =============================================================================
# SECTION 5b — HELPER: Causal Convolution Block
# =============================================================================

class CausalConv1dBlock(nn.Module):
    """
    A single causal convolution block.

    'Causal' means: at time step t, the output only depends on
    inputs at times ≤ t (no peeking into the future).

    This is achieved by padding only on the LEFT side of the sequence,
    then trimming the right side after the convolution.

    Why causal?
        Real-world speech recognition is often done in a streaming fashion —
        you process audio as it arrives. A causal model can do this naturally.
    """

    def __init__(self, conv: nn.Module, left_pad: int, activation: nn.Module):
        super().__init__()
        self.conv       = conv
        self.left_pad   = left_pad    # how many zeros to prepend
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            x: [B, C, T]  (same shape)
        """
        # Pad left side only: (left_pad zeros before time 0)
        x = F.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        x = self.activation(x)
        return x


# =============================================================================
# SECTION 6 — LOSS FUNCTIONS
# =============================================================================
#
# Three types of loss are used in GAN training:
#
#   1. ADVERSARIAL LOSS — the main GAN game
#      • Discriminator: maximise score(real) − score(fake)
#      • Generator:     maximise score(fake)  [fool the Discriminator]
#
#   2. SMOOTHNESS LOSS — penalise jittery Generator outputs
#      Encourages adjacent frames to have similar phoneme predictions.
#      Speech naturally has smooth transitions, not random jumps.
#
#   3. GRADIENT PENALTY — stabilises Discriminator training (WGAN-GP)
#      Forces the Discriminator's output to change gradually w.r.t. inputs.
#      Prevents the Discriminator from becoming too confident too fast.

def compute_generator_loss(
    fake_scores: torch.Tensor,      # [B] — Discriminator scores for fake
    fake_logprobs: torch.Tensor,    # [B, T, V] — Generator log-probabilities
    cfg: Wav2VecUConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total Generator loss.

    The Generator wants to:
        1. Produce sequences that fool the Discriminator (adversarial loss)
        2. Produce smooth transitions (smoothness penalty)

    Args:
        fake_scores:  Discriminator's scores for fake sequences
        fake_logprobs: Generator's log-probability outputs
        cfg:          Configuration

    Returns:
        total_loss:   Scalar tensor to backpropagate
        log_dict:     Dict of individual loss values for logging/monitoring
    """
    log = {}

    # ── 1. Adversarial loss ───────────────────────────────────────────────────
    #    Generator wants fake_scores to be HIGH (fool the Discriminator)
    #    Loss = −mean(score) → minimising this maximises the scores
    adv_loss = -fake_scores.mean()
    log["gen_adv_loss"] = adv_loss.item()

    total_loss = adv_loss

    # ── 2. Smoothness penalty ─────────────────────────────────────────────────
    #    Penalise large differences between adjacent frames.
    #    smooth_loss = mean( ||probs[t+1] - probs[t]||^2 )
    if cfg.smoothness_weight > 0:
        # Convert log-probs to probs, then compute frame-to-frame differences
        probs = fake_logprobs.exp()            # [B, T, V]
        diff  = probs[:, 1:, :] - probs[:, :-1, :]  # [B, T-1, V]
        smooth_loss = (diff ** 2).mean()
        log["gen_smooth_loss"] = smooth_loss.item()
        total_loss = total_loss + cfg.smoothness_weight * smooth_loss

    log["gen_total_loss"] = total_loss.item()
    return total_loss, log


def compute_discriminator_loss(
    real_scores: torch.Tensor,      # [B] — scores for real sequences
    fake_scores: torch.Tensor,      # [B] — scores for fake sequences
    real_samples: torch.Tensor,     # [B, T, C] — real data (for gradient penalty)
    fake_samples: torch.Tensor,     # [B, T, C] — fake data (for gradient penalty)
    discriminator: "Discriminator",
    cfg: Wav2VecUConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total Discriminator loss.

    The Discriminator wants:
        • real_scores to be HIGH
        • fake_scores to be LOW

    This is the classic Wasserstein loss:
        D_loss = mean(fake_scores) − mean(real_scores)
    Minimising this pushes fake_scores down and real_scores up.

    Args:
        real_scores:   Discriminator scores for real sequences
        fake_scores:   Discriminator scores for fake sequences
        real_samples:  Real input data (needed for gradient penalty)
        fake_samples:  Fake input data (needed for gradient penalty)
        discriminator: Discriminator model (needed for gradient penalty)
        cfg:           Configuration

    Returns:
        total_loss: Scalar tensor
        log_dict:   Dict of individual loss values
    """
    log = {}

    # ── 1. Wasserstein adversarial loss ───────────────────────────────────────
    #    D wants: real_scores ↑, fake_scores ↓
    #    Loss = mean(fake) − mean(real)  →  minimising this achieves the goal
    w_loss = fake_scores.mean() - real_scores.mean()
    log["disc_w_loss"]    = w_loss.item()
    log["disc_real_mean"] = real_scores.mean().item()
    log["disc_fake_mean"] = fake_scores.mean().item()

    total_loss = w_loss

    # ── 2. Gradient penalty (WGAN-GP) ─────────────────────────────────────────
    #    This regularises the Discriminator to prevent it from becoming
    #    too "sharp" (pathological gradient landscapes).
    if cfg.gradient_penalty > 0 and real_samples.requires_grad:
        gp = _gradient_penalty(discriminator, real_samples, fake_samples)
        log["disc_gp"] = gp.item()
        total_loss = total_loss + cfg.gradient_penalty * gp

    log["disc_total_loss"] = total_loss.item()
    return total_loss, log


def _gradient_penalty(
    discriminator: "Discriminator",
    real: torch.Tensor,   # [B, T, C]
    fake: torch.Tensor,   # [B, T, C]
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.

    Creates an "interpolated" sample halfway between real and fake,
    then penalises the Discriminator if its gradient norm deviates from 1.

    This enforces the Lipschitz constraint that makes WGAN training stable.

    Math:
        penalty = E[(||∇_x D(x̂)||_2 − 1)^2]
        where x̂ = α·real + (1−α)·fake,  α ~ Uniform(0,1)
    """
    # Audio frames and text phoneme rows are often different lengths; GP needs matching [B,T,C].
    if real.shape[1] != fake.shape[1]:
        T = min(real.shape[1], fake.shape[1])
        real = real[:, :T].contiguous()
        fake = fake[:, :T].contiguous()

    B, T, C = real.shape

    # Random interpolation coefficient per sample
    alpha = torch.rand(B, 1, 1, device=real.device)

    # Interpolate between real and fake
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # Get Discriminator scores for interpolated samples
    scores = discriminator(interpolated, None)

    # Compute gradients w.r.t. interpolated input
    gradients = torch.autograd.grad(
        outputs    = scores.sum(),
        inputs     = interpolated,
        create_graph = True,
        retain_graph = True,
    )[0]  # [B, T, C]

    # Gradient norm across C and T dimensions, then penalty
    grad_norm = gradients.view(B, -1).norm(2, dim=1)  # [B]
    penalty   = ((grad_norm - 1) ** 2).mean()

    return penalty


# =============================================================================
# SECTION 7 — MAIN MODEL (ties Generator + RealData + Discriminator together)
# =============================================================================

@register_model("wav2vec_u", dataclass=Wav2VecUConfig)
class Wav2VecU(BaseFairseqModel):
    """
    Wav2Vec-U: Full unsupervised speech recognition model.

    This is the top-level model class that fairseq's training loop interacts with.
    It owns all three sub-components:

        self.generator     — Generator instance
        self.real_data     — RealData instance
        self.discriminator — Discriminator instance

    The training loop alternates between:
        1. Updating the Discriminator (make it better at distinguishing real/fake)
        2. Updating the Generator (make it better at fooling the Discriminator)

    This class also handles:
        • Building the model from config
        • Device placement (CPU / MPS / CUDA)
        • Feature normalisation
        • Forward pass routing
    """

    def __init__(self, cfg: Wav2VecUConfig, target_dict):
        """
        Args:
            cfg:         Wav2VecUConfig (loaded from YAML)
            target_dict: fairseq Dictionary of phoneme symbols
        """
        super().__init__()
        self.cfg = cfg

        # ── Device selection ──────────────────────────────────────────────────
        self.device = select_device(cfg.device)

        # ── Vocabulary size ───────────────────────────────────────────────────
        #    From the target dictionary or from dataset config
        dataset_cfg      = get_active_dataset_cfg()
        num_phonemes     = (
            len(target_dict)
            if target_dict is not None
            else dataset_cfg["phoneme_vocab_size"]
        )

        # ── Feature normalisation ─────────────────────────────────────────────
        #    Instance normalisation: normalises each sample independently.
        #    This removes loudness / speaker differences from features.
        #    Think of it as "volume normalisation" on the feature space.
        feat_dim = cfg.input_dim
        # Both nn.InstanceNorm1d and nn.GroupNorm trigger MPS bugs on Apple Silicon.
        # Store affine parameters manually and compute instance norm by hand.
        # Manual ops (mean/var/mul/add) work correctly on every backend.
        self.feature_norm_weight = nn.Parameter(torch.ones(feat_dim))
        self.feature_norm_bias   = nn.Parameter(torch.zeros(feat_dim))
        # Keep a dummy attribute so the model printout stays informative
        self.feature_norm = None  # type: ignore[assignment]

        def normalize_fn(x):
            """Normalise [B, T, C] features using manual instance norm.

            Instance norm: for each (batch, channel), normalise over the T
            (time) dimension.  Input x is [B, T, C].
            """
            # Work in [B, C, T] so T is the last dim for the norm
            x_t = x.transpose(1, 2)                          # [B, C, T]
            mean = x_t.mean(dim=-1, keepdim=True)            # [B, C, 1]
            var  = x_t.var(dim=-1, keepdim=True, unbiased=False)  # [B, C, 1]
            x_t  = (x_t - mean) / (var + 1e-5).sqrt()
            # Affine transform with per-channel weight/bias  [1, C, 1]
            w = self.feature_norm_weight.view(1, -1, 1)
            b = self.feature_norm_bias.view(1, -1, 1)
            x_t = x_t * w + b
            return x_t.transpose(1, 2), None

        # ── Instantiate the three core classes ────────────────────────────────

        # 1. Generator: speech → fake phonemes
        self.generator = Generator(
            input_dim    = feat_dim,      # wav2vec 2.0 output dim (default 512)
            output_dim   = num_phonemes,
            cfg          = cfg,
            normalize_fn = normalize_fn,
        )

        # 2. RealData: text → real phoneme embeddings
        self.real_data = RealData(
            num_phonemes = num_phonemes,
            embed_dim    = cfg.discriminator_dim,  # match Discriminator's input dim
            dataset_cfg  = dataset_cfg,
        )

        # 3. Discriminator: judge real vs fake
        #    Input dim = Discriminator inner dim (real_data outputs this dim)
        self.discriminator = Discriminator(
            input_dim = cfg.discriminator_dim,
            cfg       = cfg,
        )

        # Move everything to the chosen device
        self.to(self.device)

        self.pad_idx = target_dict.pad() if target_dict is not None else 0
        self.update_num = 0

        # Fairseq composite optimizer only knows cfg.optimizer.groups keys (here:
        # generator, discriminator). Parameters without param_group are bucketed as
        # "default", which is not in the config — tag every parameter explicitly.
        for name, p in self.named_parameters():
            p.param_group = (
                "discriminator" if name.startswith("discriminator.") else "generator"
            )

        # ── Log model summary ─────────────────────────────────────────────────
        gen_params  = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        real_params = sum(p.numel() for p in self.real_data.parameters())

        logger.info("=" * 60)
        logger.info("[Wav2Vec-U] Model built successfully.")
        logger.info(f"  Device:        {self.device}")
        logger.info(f"  Phoneme vocab: {num_phonemes}")
        logger.info(f"  Generator:     {gen_params:,} parameters")
        logger.info(f"  RealData:      {real_params:,} parameters")
        logger.info(f"  Discriminator: {disc_params:,} parameters")
        logger.info(f"  Total:         {gen_params + disc_params + real_params:,} parameters")
        logger.info("=" * 60)

    @classmethod
    def build_model(cls, cfg: Wav2VecUConfig, task) -> "Wav2VecU":
        """
        Factory method called by fairseq's training infrastructure.

        fairseq uses this class method to instantiate the model from config.
        You don't call this directly — fairseq does.
        """
        target_dict = getattr(task, "target_dictionary", None)
        return cls(cfg, target_dict)

    def set_num_updates(self, num_updates: int) -> None:
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def discrim_step(self, num_updates: Optional[int] = None) -> bool:
        """Odd updates train the discriminator; even updates train the generator."""
        if num_updates is None:
            num_updates = self.update_num
        return num_updates % 2 == 1

    def get_groups_for_update(self, num_updates: int) -> str:
        return "discriminator" if self.discrim_step(num_updates) else "generator"

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        random_label: Optional[torch.Tensor] = None,
        dense_x_only: bool = False,
        aux_target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Training entry point for fairseq.

        ``random_label`` holds unpaired text phoneme ids (from RandomInputDataset).
        Alternating G/D uses ``self.update_num`` (set by ``set_num_updates``).
        """
        del aux_target  # reserved for MMI / aux losses (upstream API)
        # Infer device from model parameters at runtime rather than the
        # stale self.device value — fairseq's trainer may have moved the model
        # (e.g. from MPS to CPU) after we set self.device in __init__.
        _device = next(self.parameters()).device
        features = features.to(_device)
        if padding_mask is not None:
            padding_mask = padding_mask.to(_device)

        fake_logprobs, _fake_dense, padding_mask = self.generator(features, padding_mask)
        fake_probs = fake_logprobs.exp()
        fake_emb = torch.matmul(
            fake_probs,
            self.real_data.phoneme_embed.weight,
        )
        fake_scores = self.discriminator(fake_emb, padding_mask)

        if dense_x_only:
            return {
                "logits": fake_logprobs,
                "padding_mask": padding_mask,
            }

        real_emb = None
        real_padding_mask = None
        if random_label is not None:
            real_phonemes = random_label.to(_device)
            real_padding_mask = real_phonemes.eq(self.pad_idx)
            real_emb = self.real_data(real_phonemes, real_padding_mask)

        d_step = self.discrim_step(self.update_num)
        sample_size = features.size(0)

        result: Dict[str, Any] = {
            "fake_logprobs": fake_logprobs,
            "fake_scores": fake_scores,
            "sample_size": sample_size,
            "temp": torch.tensor(2.0, device=features.device),
            "code_ppl": torch.tensor(0.0, device=features.device),
        }

        if d_step:
            if real_emb is None:
                raise RuntimeError(
                    "Discriminator step requires unpaired text phonemes (random_label)."
                )
            real_scores = self.discriminator(real_emb, real_padding_mask)
            disc_loss, disc_log = compute_discriminator_loss(
                real_scores=real_scores,
                fake_scores=fake_scores,
                real_samples=real_emb,
                fake_samples=fake_emb.detach(),
                discriminator=self.discriminator,
                cfg=self.cfg,
            )
            result["losses"] = {"dense_d": disc_loss}
            result["real_scores"] = real_scores
            result["logs"] = disc_log
        else:
            gen_loss, gen_log = compute_generator_loss(
                fake_scores, fake_logprobs, self.cfg
            )
            result["losses"] = {"dense_g": gen_loss}
            result["logs"] = gen_log

        return result

    def get_normalized_probs(
        self,
        net_output: Any,
        log_probs: bool,
        sample: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Required by fairseq for CTC / evaluation decoding.

        Returns the Generator's output probabilities,
        formatted as expected by fairseq's decoding utilities.
        """
        if isinstance(net_output, dict):
            logits = net_output["logits"]
        else:
            logits = net_output[0]  # [B, T, V] — Generator's log-probs
        if log_probs:
            return logits  # already log_softmax
        else:
            return logits.exp()


# =============================================================================
# SECTION 8 — TRAINING LOOP (standalone demo, not used by fairseq)
# =============================================================================
#
# This section lets you run and observe the GAN training directly,
# WITHOUT needing the full fairseq infrastructure.
#
# It demonstrates:
#   • How data flows through Generator → Discriminator
#   • How losses are computed and backpropagated
#   • Progress bars for each training step
#
# Run from command line:
#   python wav2vec_u.py
#
# This is great for understanding the code before plugging into fairseq.

def demo_training_loop(
    num_steps:  int = 20,
    batch_size: int = 4,
    seq_len:    int = 50,
    text_len:   int = 30,
    device_pref: str = "auto",
):
    """
    Standalone demo: trains Generator and Discriminator on random data.

    This is NOT the real training loop (fairseq handles that).
    It's a self-contained demonstration so you can see the code run
    and understand how the three classes interact.

    Args:
        num_steps:   Number of GAN training iterations to run
        batch_size:  Samples per mini-batch
        seq_len:     Audio feature sequence length (frames)
        text_len:    Text phoneme sequence length (frames)
        device_pref: "auto" | "cpu" | "mps" | "cuda"
    """

    print("\n" + "=" * 60)
    print("  Wav2Vec-U  —  GAN Demo Training Loop")
    print("=" * 60)
    print("  This demo runs the Generator + Discriminator on random")
    print("  data so you can verify the code works before using")
    print("  real LibriSpeech data with the full fairseq pipeline.")
    print("=" * 60 + "\n")

    # ── Device ────────────────────────────────────────────────────────────────
    device = select_device(device_pref)
    print(f"  Device:     {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_len} frames (audio)  /  {text_len} frames (text)")
    print(f"  GAN steps:  {num_steps}\n")

    # ── Dummy configuration ───────────────────────────────────────────────────
    cfg = Wav2VecUConfig()     # uses all default hyperparameter values
    cfg.device = device_pref

    VOCAB = 41  # standard English phoneme count

    # ── Instantiate the three core classes ────────────────────────────────────
    print("  Instantiating Generator, RealData, Discriminator...")

    generator     = Generator(input_dim=512, output_dim=VOCAB, cfg=cfg).to(device)
    real_data_mod = RealData(num_phonemes=VOCAB, embed_dim=cfg.discriminator_dim).to(device)
    discriminator = Discriminator(input_dim=cfg.discriminator_dim, cfg=cfg).to(device)

    print("  ✓ All three classes ready.\n")

    # ── Optimisers ────────────────────────────────────────────────────────────
    #    Separate optimisers for Generator and Discriminator.
    #    They are updated alternately in the training loop.
    gen_opt  = torch.optim.Adam(generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print(f"  {'Step':>5}  {'D Loss':>10}  {'G Loss':>10}  {'D(real)':>10}  {'D(fake)':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in tqdm(range(1, num_steps + 1), desc="  GAN Training", unit="step"):

        # ── Generate synthetic batch (replaces real data loading) ──────────────
        # In real training, these come from the fairseq data loader:
        #   • speech_features: wav2vec 2.0 embeddings of audio clips
        #   • phoneme_ids:     integer indices from text phonemisation pipeline

        speech_features = torch.randn(batch_size, seq_len, 512, device=device)
        phoneme_ids     = torch.randint(0, VOCAB, (batch_size, text_len), device=device)

        # =======================================================================
        # DISCRIMINATOR UPDATE STEP
        # =======================================================================
        # Goal: make Discriminator better at telling real from fake.

        disc_opt.zero_grad()

        # Get fake sequences from Generator (no Generator gradient needed here)
        with torch.no_grad():
            fake_logprobs, _, _ = generator(speech_features)
        fake_probs = fake_logprobs.exp()
        fake_emb   = torch.matmul(fake_probs, real_data_mod.phoneme_embed.weight)

        # Get real embeddings
        real_emb = real_data_mod(phoneme_ids)

        # Score both
        real_scores  = discriminator(real_emb)
        fake_scores  = discriminator(fake_emb.detach())

        # Wasserstein loss: push real up, fake down
        disc_loss = fake_scores.mean() - real_scores.mean()
        disc_loss.backward()
        disc_opt.step()

        # =======================================================================
        # GENERATOR UPDATE STEP
        # =======================================================================
        # Goal: make Generator better at fooling Discriminator.

        gen_opt.zero_grad()

        fake_logprobs, _, _ = generator(speech_features)
        fake_probs       = fake_logprobs.exp()
        fake_emb         = torch.matmul(fake_probs, real_data_mod.phoneme_embed.weight)
        fake_scores_gen  = discriminator(fake_emb)

        # Generator loss: wants fake_scores to be HIGH → negate
        gen_loss = -fake_scores_gen.mean()
        gen_loss.backward()
        gen_opt.step()

        # ── Log progress every 5 steps ─────────────────────────────────────────
        if step % 5 == 0 or step == 1:
            tqdm.write(
                f"  {step:>5}  "
                f"{disc_loss.item():>10.4f}  "
                f"{gen_loss.item():>10.4f}  "
                f"{real_scores.mean().item():>10.4f}  "
                f"{fake_scores.mean().item():>10.4f}"
            )

    print("\n" + "=" * 60)
    print("  Demo complete! The GAN trained for", num_steps, "steps.")
    print("\n  What just happened:")
    print("  • Generator took random 'speech features' and produced")
    print("    fake phoneme probability sequences.")
    print("  • Discriminator tried to distinguish those from real")
    print("    phoneme sequences (from the RealData module).")
    print("  • Both improved together — this is the GAN game!")
    print("\n  In real Wav2Vec-U training:")
    print("  • Speech features come from the wav2vec 2.0 encoder.")
    print("  • Real phonemes come from the LibriSpeech text corpus.")
    print("  • Training runs for thousands of steps on a GPU.")
    print("=" * 60 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Set up logging so you can see info messages
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(levelname)s | %(name)s | %(message)s",
        stream  = sys.stdout,
    )

    # ── Quick sanity test ─────────────────────────────────────────────────────
    print("\n[Sanity Check] Verifying all three classes instantiate correctly...\n")

    test_cfg = Wav2VecUConfig()

    gen   = Generator(input_dim=512, output_dim=41, cfg=test_cfg)
    rd    = RealData(num_phonemes=41, embed_dim=test_cfg.discriminator_dim)
    disc  = Discriminator(input_dim=test_cfg.discriminator_dim, cfg=test_cfg)

    # Quick forward pass with dummy data
    B, T, C = 2, 30, 512
    dummy_speech  = torch.randn(B, T, C)
    dummy_phones  = torch.randint(0, 41, (B, 20))

    fake_out, _    = gen(dummy_speech)
    real_out       = rd(dummy_phones)
    fake_probs     = fake_out.exp()
    fake_emb       = torch.matmul(fake_probs, rd.phoneme_embed.weight)
    disc_real      = disc(real_out)
    disc_fake      = disc(fake_emb)

    print(f"  Generator output shape:     {fake_out.shape}   ← [B={B}, T={T}, V=41]")
    print(f"  RealData output shape:      {real_out.shape}   ← [B={B}, T=20, embed=256]")
    print(f"  Discriminator real scores:  {disc_real}  ← [B={B}]")
    print(f"  Discriminator fake scores:  {disc_fake}  ← [B={B}]")
    print("\n  ✓ All classes working correctly!\n")

    # ── Run the demo training loop ────────────────────────────────────────────
    # Change device_pref to "mps" for Apple Silicon GPU
    demo_training_loop(
        num_steps   = 20,
        batch_size  = 4,
        seq_len     = 50,
        text_len    = 30,
        device_pref = "auto",   # ← change to "mps", "cuda", or "cpu"
    )
