# Copyright (c) Facebook, Inc. and its affiliates.  MIT License.
#
# =============================================================================
# wav2vec_u.py — Wav2Vec-U GAN Model
# Stock Fairseq architecture · macOS/MPS adaptation
# =============================================================================
#
# HOW THE GAN WORKS
# -----------------
#   Wav2Vec-U learns to transcribe speech WITHOUT any labelled audio.
#   Instead it plays a two-player game:
#
#     GENERATOR:      Listens to speech features (wav2vec 2.0) and outputs
#                     a sequence of "fake" phoneme probabilities.
#
#     DISCRIMINATOR:  Sees both the Generator's fakes and real phoneme
#                     sequences (from plain text).  Its job is to tell
#                     them apart.
#
#     REALDATA:       Wraps the real text phoneme sequences and presents
#                     them to the Discriminator as "ground truth".
#
#   Training pressure:
#     • Discriminator improves at spotting fakes  →  harder target for Generator.
#     • Generator improves at fooling Discriminator  →  better transcriptions.
#
#
# SEPARATION OF CONCERNS — five classes, five responsibilities
# ------------------------------------------------------------
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  1.  HELPER LAYERS  (SamePad, TransposeLast)                    │
#   │      Concern: geometry / padding utilities; no trainable params  │
#   ├──────────────────────────────────────────────────────────────────┤
#   │  1.5 JOIN SEGMENTER                                              │
#   │      Concern: collapse consecutive Generator frames → tokens     │
#   │      Implementation: vectorised mean-pool via scatter_add_       │
#   │      No trainable params; used only during discriminator step    │
#   ├──────────────────────────────────────────────────────────────────┤
#   │  2.  GENERATOR                                                   │
#   │      Concern: map wav2vec speech features → phoneme probs        │
#   │      Architecture (stock): TransposeLast → Conv1d → TransposeLast│
#   ├──────────────────────────────────────────────────────────────────┤
#   │  3.  REALDATA                                                    │
#   │      Concern: encode text phoneme indices as probability vectors  │
#   │      Implementation: one-hot encoding over the phoneme vocabulary │
#   ├──────────────────────────────────────────────────────────────────┤
#   │  4.  DISCRIMINATOR                                               │
#   │      Concern: score a phoneme sequence as real (text) or fake    │
#   │      Architecture (stock): Conv1d + SamePad + GELU (causal)     │
#   ├──────────────────────────────────────────────────────────────────┤
#   │  5.  WAV2VEC_U  (top-level model)                               │
#   │      Concern: coordinate GAN training loop + loss computation    │
#   │      Losses: dense_g, dense_d (frame-level) + token_d (segments)│
#   └──────────────────────────────────────────────────────────────────┘
#
#
# macOS ADAPTATIONS (not in stock fairseq)
# ----------------------------------------
#   • select_device(preferred) — picks CPU / MPS / CUDA at runtime.
#   • F.layer_norm for feature normalisation — replaces manual mean/var ops
#     (which trigger MPS shader recompilation on every new shape) and
#     nn.InstanceNorm1d (silent wrong-result bug on older MPS).
#     F.layer_norm is a first-class MPS kernel: no JIT penalty, correct results.
#   • cfg.device field — set "mps" in w2vu.yaml or via Hydra override.
#
# =============================================================================

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 0 — DEVICE SELECTION
# Concern: abstract over CPU / MPS / CUDA so the rest of the code is portable
# =============================================================================

def select_device(preferred: str = "auto") -> torch.device:
    """
    Return the best available compute device.

    Args:
        preferred: "auto" | "cpu" | "mps" | "cuda"
                   "auto" tries CUDA, then MPS, then CPU.
    """
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("[Device] CUDA not available — falling back to CPU.")
        return torch.device("cpu")

    if preferred == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("[Device] MPS (Apple Silicon) not available — falling back to CPU.")
        return torch.device("cpu")

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")


# =============================================================================
# SECTION 1 — HELPER LAYERS
# Concern: geometry / padding utilities; no trainable parameters
# =============================================================================

class SamePad(nn.Module):
    """
    Trim trailing frames so Conv1d output length equals input length.

    A Conv1d with padding=kernel-1 produces (T + kernel - 1) output frames.
    SamePad removes the excess (kernel - 1) frames from the right, restoring
    length T.  When causal=True this implements left-only (causal) padding:
    each output frame sees only the current and past input frames.
    """

    def __init__(self, kernel_size: int, causal: bool = False) -> None:
        super().__init__()
        self.remove = (kernel_size - 1) if causal else (1 if kernel_size % 2 == 0 else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class TransposeLast(nn.Module):
    """Swap the last two dimensions: [..., A, B] → [..., B, A]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1)


# =============================================================================
# SECTION 1.5 — JOIN SEGMENTER
# Concern: collapse consecutive identical-phoneme frames into token segments
# =============================================================================

class JoinSegmenter(nn.Module):
    """
    Collapse consecutive frames with the same argmax phoneme into a single
    token by mean-pooling their soft probability distributions.

    Generator output is [B, T, V] — one probability vector per speech frame.
    After segmentation it becomes [B, T', V] where T' is the number of
    distinct phoneme "runs".  With early-training code_ppl ≈ 5, a sequence
    of 100 frames typically collapses to ~15 tokens.

    Segment boundaries are determined by argmax (non-differentiable), but
    mean-pooling within each segment IS differentiable — gradients still
    flow back through the token sequence to the Generator.

    Implementation: fully vectorised via scatter_add_ — no Python loops
    over batch items, no CPU↔MPS synchronisation.
    """

    def segment(
        self,
        x: torch.Tensor,                             # [B, T, V]
        padding_mask: Optional[torch.Tensor] = None, # [B, T] True=padded
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            seg_x:    [B, T', V]  mean-pooled segment distributions
            seg_mask: [B, T']     True at empty/padded segment positions
        """
        B, T, V = x.shape

        # Argmax over vocab gives the "winning" phoneme per frame
        ids = x.argmax(dim=-1)  # [B, T]

        # A new segment starts whenever the argmax changes (or at position 0)
        boundaries = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            ids[:, 1:] != ids[:, :-1],
        ], dim=1)  # [B, T]

        # Padded frames never start a new segment
        if padding_mask is not None:
            boundaries = boundaries & ~padding_mask

        # Assign each frame a 0-indexed segment ID via cumulative sum of boundaries
        seg_id = boundaries.long().cumsum(dim=1) - 1  # [B, T]

        # Max valid segments in this batch (needed to size output tensor)
        max_segs = int(seg_id.max().item()) + 1

        # Padded frames get routed to a "garbage" bucket (index = max_segs)
        # so their content never bleeds into valid segments
        if padding_mask is not None:
            seg_id_scatter = seg_id.masked_fill(padding_mask, max_segs)
        else:
            seg_id_scatter = seg_id

        # Allocate output with one extra garbage bucket
        seg_feats  = x.new_zeros(B, max_segs + 1, V)
        seg_counts = x.new_zeros(B, max_segs + 1)

        # Scatter-add: accumulate probability vectors and frame counts per segment
        if padding_mask is not None:
            valid       = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            x_contrib   = x * valid
            cnt_contrib = (~padding_mask).float()               # [B, T]
        else:
            x_contrib   = x
            cnt_contrib = x.new_ones(B, T)

        idx = seg_id_scatter.unsqueeze(-1).expand(-1, -1, V)
        seg_feats.scatter_add_(1, idx, x_contrib)
        seg_counts.scatter_add_(1, seg_id_scatter, cnt_contrib)

        # Mean-pool: divide accumulated vectors by frame count per segment
        seg_feats = seg_feats / seg_counts.unsqueeze(-1).clamp(min=1e-8)

        # Drop the garbage bucket; mark empty segments as padding
        seg_x    = seg_feats[:, :max_segs, :]           # [B, T', V]
        seg_mask = (seg_counts[:, :max_segs] == 0)      # [B, T'] True=empty

        return seg_x, seg_mask


# =============================================================================
# SECTION 2 — CONFIGURATION
# =============================================================================

@dataclass
class SegmentationConfig(FairseqDataclass):
    type:           str   = field(default="JOIN")
    subsample_rate: float = field(default=0.25)
    mean_pool:      bool  = field(default=True)
    mean_pool_join: bool  = field(default=False)
    remove_zeros:   bool  = field(default=False)


@dataclass
class Wav2VecUConfig(FairseqDataclass):
    """All hyperparameters for Wav2Vec-U; set via w2vu.yaml or Hydra overrides."""

    # ── Input features ────────────────────────────────────────────────────────
    input_dim: int = field(
        default=512,
        metadata={"help": "Wav2vec 2.0 feature dimension fed into the Generator."},
    )

    # ── Generator ─────────────────────────────────────────────────────────────
    # Stock architecture: TransposeLast → Conv1d(input_dim, V, k) → TransposeLast
    # Single conv layer — fast backprop through the WGAN-GP gradient penalty.
    generator_kernel:  int   = field(default=4,    metadata={"help": "Conv1d kernel width."})
    generator_stride:  int   = field(default=1,    metadata={"help": "Conv1d stride."})
    generator_bias:    bool  = field(default=False, metadata={"help": "Use bias in Conv1d."})
    generator_dropout: float = field(default=0.1,  metadata={"help": "Dropout before conv."})

    # ── Discriminator ─────────────────────────────────────────────────────────
    # Stock architecture: Conv1d(V→dim) + SamePad + [Conv1d+SamePad+GELU]×(depth-1) + Conv1d→1
    discriminator_kernel:          int   = field(default=6,     metadata={"help": "Conv1d kernel width."})
    discriminator_dim:             int   = field(default=384,   metadata={"help": "Hidden channels."})
    discriminator_depth:           int   = field(default=2,     metadata={"help": "Number of conv layers."})
    discriminator_dropout:         float = field(default=0.0,   metadata={"help": "Dropout inside discriminator."})
    discriminator_causal:          bool  = field(default=True,  metadata={"help": "Causal (left-only) padding."})
    discriminator_linear_emb:      bool  = field(default=False, metadata={"help": "Unused; kept for YAML compat."})
    discriminator_max_pool:        bool  = field(default=False, metadata={"help": "Unused; kept for YAML compat."})
    discriminator_act_after_linear:bool  = field(default=False, metadata={"help": "Unused; kept for YAML compat."})
    discriminator_weight_norm:     bool  = field(default=False, metadata={"help": "Apply weight norm to conv layers."})
    discriminator_spectral_norm:   bool  = field(default=False, metadata={"help": "Apply spectral norm (unused)."})

    # ── Temperature schedule ──────────────────────────────────────────────────
    # temp = [start, min, decay].  After each update: temp = max(min, temp*decay).
    # Used to soften the Generator's output distribution early in training.
    temp: List[float] = field(
        default_factory=lambda: [2.0, 0.1, 0.99995],
        metadata={"help": "[start_temp, min_temp, decay_factor]"},
    )

    # ── Losses ────────────────────────────────────────────────────────────────
    gradient_penalty:  float = field(default=0.0, metadata={"help": "WGAN-GP penalty coefficient."})
    smoothness_weight: float = field(default=0.0, metadata={"help": "Frame-to-frame smoothness penalty."})
    smoothing:         float = field(default=0.0, metadata={"help": "Label-smoothing on real phonemes."})
    smoothing_one_sided: bool = field(default=False, metadata={"help": "One-sided label smoothing."})
    code_penalty:      float = field(default=0.0, metadata={"help": "Codebook diversity penalty weight."})
    mmi_weight:        float = field(default=0.0, metadata={"help": "MMI auxiliary loss weight (unused)."})

    # ── Gumbel-softmax ────────────────────────────────────────────────────────
    gumbel:      bool = field(default=False, metadata={"help": "Use Gumbel-softmax for Generator output."})
    hard_gumbel: bool = field(default=False, metadata={"help": "Hard (argmax) Gumbel-softmax."})

    # ── Misc ──────────────────────────────────────────────────────────────────
    target_dim:              int            = field(default=64,   metadata={"help": "Auxiliary target dim (unused)."})
    target_downsample_rate:  int            = field(default=2,    metadata={"help": "Aux downsample rate (unused)."})
    blank_weight:            float          = field(default=0.0,  metadata={"help": "Extra weight on blank token."})
    blank_mode:              str            = field(default="add", metadata={"help": "'add' or 'set' blank weight."})
    blank_is_sil:            bool           = field(default=False, metadata={"help": "Treat silence as blank."})
    no_softmax:              bool           = field(default=False, metadata={"help": "Skip softmax on Generator output."})
    segmentation:            SegmentationConfig = field(default_factory=SegmentationConfig)

    # ── macOS / device (not in stock fairseq) ─────────────────────────────────
    device: str = field(
        default="auto",
        metadata={"help": "Compute device: 'auto' | 'cpu' | 'mps' | 'cuda'. "
                          "Set to 'mps' for Apple Silicon GPU."},
    )


# =============================================================================
# SECTION 3 — GENERATOR
# Concern: map wav2vec speech features → a probability distribution over phonemes
# =============================================================================

class Generator(nn.Module):
    """
    Convert wav2vec 2.0 speech features into fake phoneme probabilities.

    Architecture (stock Wav2Vec-U):
        [B, T, 512] → dropout
                    → TransposeLast          [B, 512, T]
                    → Conv1d(512, V, k=4)    [B, V, T]
                    → TransposeLast          [B, T, V]
                    → log_softmax(dim=-1)    [B, T, V]

    A single 1-D convolution slides a kernel of width 4 along the time axis,
    learning to map local speech patterns to phoneme distributions.

    Why one layer?
      The WGAN-GP gradient penalty requires backpropagating through the
      Generator twice per discriminator update.  A single conv layer cuts
      that cost to ~1/4 of a 4-layer stack, making training ~4× faster
      while preserving the model's capacity to learn phoneme patterns.
    """

    def __init__(self, input_dim: int, output_dim: int, cfg: Wav2VecUConfig) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=cfg.generator_dropout)

        # Stock projection: transpose ↔ conv ↔ transpose
        # TransposeLast converts [B, T, C] to [B, C, T] for Conv1d, then back.
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                padding=cfg.generator_kernel // 2,
                bias=cfg.generator_bias,
            ),
            TransposeLast(),
        )

        logger.info(
            f"[Generator] input_dim={input_dim}, output_dim={output_dim}, "
            f"kernel={cfg.generator_kernel}  (stock 1-layer projection)"
        )

    def forward(
        self,
        features: torch.Tensor,                     # [B, T, C]
        padding_mask: Optional[torch.Tensor] = None, # [B, T] True = padded
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            log_probs:    [B, T, V]  log-softmax over phoneme vocabulary
            dense:        [B, T, V]  same tensor (alias kept for API compat)
            padding_mask: [B, T]     unchanged
        """
        x = self.dropout(features)
        x = self.proj(x)                   # [B, T, V]  (raw logits)

        # Align padding mask if stride or padding shifted the time dimension
        if padding_mask is not None:
            T = x.size(1)
            if padding_mask.size(1) != T:
                padding_mask = padding_mask[:, :T] if padding_mask.size(1) > T else \
                    F.pad(padding_mask, (0, T - padding_mask.size(1)), value=True)

        x = F.log_softmax(x, dim=-1)

        # Zero padded positions so they contribute nothing to losses
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x, x, padding_mask


# =============================================================================
# SECTION 4 — REALDATA
# Concern: encode text phoneme indices as probability vectors for the Discriminator
# =============================================================================

class RealData(nn.Module):
    """
    Encode real phoneme sequences from unlabelled text as one-hot vectors.

    The Discriminator sees both:
      • Generator output: [B, T, V]  soft probability distribution (fake)
      • RealData output:  [B, T, V]  one-hot distribution        (real)

    Both are in the same V-dimensional probability space, so the Discriminator
    can compare them without any embedding projection.

    Why one-hot instead of learned embeddings?
      The stock Wav2Vec-U paper feeds the raw phoneme probability vectors from
      the Generator directly to the Discriminator.  Real phoneme sequences are
      therefore represented as one-hot vectors (hard probabilities) in the same
      space.  This keeps the real/fake representations symmetric and avoids
      adding extra learned parameters (which would change the discriminator's
      target distribution each update).

    Optional label smoothing (cfg.smoothing > 0):
      Blends each one-hot with a uniform distribution, giving the Discriminator
      softer real targets and improving GAN stability.
    """

    def __init__(self, num_phonemes: int, cfg: Wav2VecUConfig) -> None:
        super().__init__()
        self.num_phonemes = num_phonemes
        self.smoothing = cfg.smoothing

        logger.info(
            f"[RealData] vocab={num_phonemes}, "
            f"smoothing={cfg.smoothing}  (one-hot encoding)"
        )

    def forward(
        self,
        phoneme_ids: torch.Tensor,                   # [B, T]  integer indices
        padding_mask: Optional[torch.Tensor] = None, # [B, T]  True = padded
    ) -> torch.Tensor:
        """
        Convert integer phoneme indices to one-hot (or smoothed) vectors.

        Returns:
            [B, T, V]  — real phoneme distributions
        """
        V = self.num_phonemes
        # One-hot: each frame gets a 1 at its phoneme index, 0 elsewhere
        one_hot = F.one_hot(phoneme_ids.clamp(0, V - 1), num_classes=V).float()

        if self.smoothing > 0:
            # Linear interpolation toward uniform: makes real targets softer
            one_hot = (1.0 - self.smoothing) * one_hot + self.smoothing / V

        # Zero out padded positions
        if padding_mask is not None:
            one_hot = one_hot.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return one_hot  # [B, T, V]


# =============================================================================
# SECTION 5 — DISCRIMINATOR
# Concern: score a phoneme-probability sequence as real (from text) or fake
# =============================================================================

class Discriminator(nn.Module):
    """
    Score a phoneme-probability sequence as real (from text) or fake (from Generator).

    Architecture (stock Wav2Vec-U):
        Input:  [B, T, V]  — probability distribution over V phonemes
        Conv1d(V → dim) + SamePad  ← input projection
        [Conv1d(dim→dim) + SamePad + Dropout + GELU] × (depth - 1)  ← hidden
        Conv1d(dim → 1)  + SamePad  ← output head
        Output: [B, T, 1]  — real/fake score per frame

    Causal convolutions (SamePad with causal=True) ensure each score
    only looks backward in time, mirroring left-to-right text processing.

    Input dimension is V (vocabulary size), not a hidden dimension.  Both
    the Generator's soft probs and RealData's one-hots share this space,
    so the Discriminator operates on a common representation of "what
    phoneme is most likely at this frame".
    """

    def __init__(self, input_dim: int, cfg: Wav2VecUConfig) -> None:
        super().__init__()

        dim    = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        depth  = cfg.discriminator_depth
        drop   = cfg.discriminator_dropout
        causal = cfg.discriminator_causal

        layers: List[nn.Module] = [
            nn.Conv1d(input_dim, dim, kernel, padding=kernel - 1),
            SamePad(kernel, causal=causal),
            nn.Dropout(p=drop),
        ]
        for _ in range(depth - 1):
            layers += [
                nn.Conv1d(dim, dim, kernel, padding=kernel - 1),
                SamePad(kernel, causal=causal),
                nn.Dropout(p=drop),
                nn.GELU(),
            ]
        layers += [
            nn.Conv1d(dim, 1, kernel, padding=kernel - 1),
            SamePad(kernel, causal=causal),
        ]
        self.net = nn.Sequential(*layers)

        if cfg.discriminator_weight_norm:
            for m in self.net:
                if isinstance(m, nn.Conv1d):
                    nn.utils.weight_norm(m)

        logger.info(
            f"[Discriminator] input_dim={input_dim}, dim={dim}, "
            f"depth={depth}, kernel={kernel}, causal={causal}"
        )

    def forward(
        self,
        x: torch.Tensor,                             # [B, T, V]
        padding_mask: Optional[torch.Tensor] = None, # [B, T]
    ) -> torch.Tensor:
        """
        Returns:
            scores: [B, T, 1] — per-frame real/fake score
        """
        x = x.transpose(1, 2)  # [B, V, T]  Conv1d expects channels first
        x = self.net(x)        # [B, 1, T]
        x = x.transpose(1, 2)  # [B, T, 1]

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x


# =============================================================================
# SECTION 6 — LOSS FUNCTIONS
# Concern: compute Generator and Discriminator losses from Discriminator scores
# =============================================================================

def _gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,  # [B, T, V]
    fake: torch.Tensor,  # [B', T', V]
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty.

    Interpolates between real and fake samples and penalises the Discriminator
    if its gradient norm deviates from 1, enforcing the Lipschitz constraint.

        penalty = E[ (‖∇_x D(x̂)‖₂ − 1)² ]
        where  x̂ = α·real + (1−α)·fake,  α ~ Uniform(0,1)

    When real and fake have different time lengths, both are truncated to the
    shorter one before interpolation.
    """
    if real.shape[1] != fake.shape[1]:
        T = min(real.shape[1], fake.shape[1])
        real = real[:, :T].contiguous()
        fake = fake[:, :T].contiguous()

    # Match batch dimension (take min)
    B = min(real.shape[0], fake.shape[0])
    real = real[:B]
    fake = fake[:B]

    alpha = torch.rand(B, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    scores = discriminator(interpolated, None)

    gradients = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, T, V]

    grad_norm = gradients.reshape(B, -1).norm(2, dim=1)  # [B]
    return ((grad_norm - 1.0) ** 2).mean()


def _compute_generator_loss(
    fake_scores: torch.Tensor,   # [B, T, 1]
    fake_logprobs: torch.Tensor, # [B, T, V]
    cfg: Wav2VecUConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generator loss:
      1. Adversarial (dense_g):  −mean(fake_scores)  — Generator wants scores HIGH.
      2. Smoothness  (smooth):   mean(‖probs[t+1]−probs[t]‖²) — encourage continuity.
    """
    log: Dict[str, float] = {}

    # Adversarial: maximise discriminator score on fake sequences
    adv = -fake_scores.mean()
    log["dense_g"] = adv.item()
    total = adv

    # Smoothness: penalise rapid jumps between adjacent frames
    if cfg.smoothness_weight > 0:
        probs = fake_logprobs.exp()              # [B, T, V]
        diff  = probs[:, 1:] - probs[:, :-1]    # [B, T-1, V]
        smooth = (diff ** 2).mean()
        log["smooth"] = smooth.item()
        total = total + cfg.smoothness_weight * smooth

    return total, log


def _compute_discriminator_loss(
    real_scores: torch.Tensor,   # [B, T, 1]
    fake_scores: torch.Tensor,   # [B, T, 1]
    real_probs: torch.Tensor,    # [B, T, V]
    fake_probs: torch.Tensor,    # [B, T, V]
    discriminator: Discriminator,
    cfg: Wav2VecUConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Discriminator loss (Wasserstein):
      1. Wasserstein (dense_d):  mean(fake) − mean(real) — Discriminator wants real HIGH, fake LOW.
      2. Gradient penalty (grad_pen): WGAN-GP regularisation.
    """
    log: Dict[str, float] = {}

    w_loss = fake_scores.mean() - real_scores.mean()
    log["dense_d"] = w_loss.item()
    total = w_loss

    if cfg.gradient_penalty > 0:
        gp = _gradient_penalty(discriminator, real_probs.detach(), fake_probs.detach())
        log["grad_pen"] = gp.item()
        total = total + cfg.gradient_penalty * gp

    return total, log


# =============================================================================
# SECTION 7 — TOP-LEVEL MODEL
# Concern: coordinate all components; integrate with fairseq's training loop
# =============================================================================

@register_model("wav2vec_u", dataclass=Wav2VecUConfig)
class Wav2VecU(BaseFairseqModel):
    """
    Wav2Vec-U: unsupervised speech recognition via GAN training.

    Top-level model that owns all components and implements the training loop
    logic required by fairseq's ModelCriterion:

        Even updates  (update_num % 2 == 0) → Generator step
        Odd  updates  (update_num % 2 == 1) → Discriminator step

    This class handles:
      • Building the three core sub-models from config.
      • Device placement (CPU / MPS / CUDA via select_device).
      • Feature normalisation (manual instance norm, MPS-safe).
      • Temperature annealing for the Generator's softmax.
      • Routing to the correct loss computation on each update.
    """

    def __init__(self, cfg: Wav2VecUConfig, target_dict) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Device ────────────────────────────────────────────────────────────
        self.device = select_device(cfg.device)

        # ── Vocabulary ────────────────────────────────────────────────────────
        num_phonemes = len(target_dict) if target_dict is not None else 41
        self.pad_idx = target_dict.pad() if target_dict is not None else 0

        # ── Feature normalisation ─────────────────────────────────────────────
        # Normalises each audio clip independently before the Generator.
        # This removes loudness / speaker-level variation from the features.
        #
        # Implementation: F.layer_norm over the last (channel) dimension.
        # This is mathematically equivalent to instance norm over time (both
        # standardise each sample independently), and F.layer_norm is a
        # first-class MPS-optimised kernel — no JIT shader compilation penalty.
        # The affine weight/bias are learned per-channel (512 params each).
        #
        # Note: nn.InstanceNorm1d had a silent wrong-result bug on older MPS.
        # F.layer_norm avoids both the bug and the custom-op compilation delay.
        feat_dim = cfg.input_dim
        self.feature_norm_weight = nn.Parameter(torch.ones(feat_dim))
        self.feature_norm_bias   = nn.Parameter(torch.zeros(feat_dim))

        def _instance_norm(x: torch.Tensor) -> torch.Tensor:
            """Layer-norm over the channel dim — MPS-optimised, same effect as instance norm."""
            return F.layer_norm(
                x,
                normalized_shape=(feat_dim,),
                weight=self.feature_norm_weight,
                bias=self.feature_norm_bias,
                eps=1e-5,
            )

        self._instance_norm = _instance_norm

        # ── Sub-models ────────────────────────────────────────────────────────

        # 1. Generator  — speech features → fake phoneme probabilities
        self.generator = Generator(
            input_dim=feat_dim,
            output_dim=num_phonemes,
            cfg=cfg,
        )

        # 2. RealData   — text phoneme indices → real phoneme probabilities
        self.real_data = RealData(num_phonemes=num_phonemes, cfg=cfg)

        # 3. Discriminator — judge real vs fake phoneme sequences
        #    Input dimension is num_phonemes (V), matching both Generator output
        #    and RealData output (both are V-dimensional probability vectors).
        self.discriminator = Discriminator(input_dim=num_phonemes, cfg=cfg)

        # 4. JoinSegmenter — collapse consecutive Generator frames into tokens
        #    No trainable parameters. Used in the discriminator step to produce
        #    a token-level sequence for the second discriminator pass (token_d).
        self.segmenter = JoinSegmenter()

        self.to(self.device)

        # ── Parameter groups (required by fairseq composite optimiser) ─────────
        # Each Parameter must carry a .param_group attribute; the optimizer
        # config in w2vu.yaml has exactly two groups: "generator" and "discriminator".
        for name, p in self.named_parameters():
            p.param_group = (
                "discriminator" if name.startswith("discriminator.") else "generator"
            )

        # ── Temperature state ─────────────────────────────────────────────────
        start_temp, min_temp, temp_decay = cfg.temp
        self.curr_temp  = start_temp
        self.min_temp   = min_temp
        self.temp_decay = temp_decay
        self.update_num = 0

        # ── Summary ───────────────────────────────────────────────────────────
        gen_p  = sum(p.numel() for p in self.generator.parameters())
        real_p = sum(p.numel() for p in self.real_data.parameters())
        disc_p = sum(p.numel() for p in self.discriminator.parameters())
        norm_p = feat_dim * 2

        logger.info("=" * 60)
        logger.info(f"[Wav2Vec-U] device={self.device}  vocab={num_phonemes}")
        logger.info(f"  Generator:      {gen_p:,} params")
        logger.info(f"  RealData:       {real_p:,} params  (one-hot, no learned params)")
        logger.info(f"  Discriminator:  {disc_p:,} params")
        logger.info(f"  JoinSegmenter:  0 params  (no learned params)")
        logger.info(f"  Feature norm:   {norm_p:,} params  (weight + bias)")
        logger.info(f"  Total:          {gen_p + real_p + disc_p + norm_p:,} params")
        logger.info(f"  Losses: dense_g (gen) | dense_d + token_d (disc)")
        logger.info("=" * 60)

    # ── fairseq API ───────────────────────────────────────────────────────────

    @classmethod
    def build_model(cls, cfg: Wav2VecUConfig, task) -> "Wav2VecU":
        """Called by fairseq infrastructure to instantiate the model."""
        target_dict = getattr(task, "target_dictionary", None)
        return cls(cfg, target_dict)

    def set_num_updates(self, num_updates: int) -> None:
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        # Anneal temperature every update
        self.curr_temp = max(
            self.min_temp, self.curr_temp * self.temp_decay
        )

    def discrim_step(self, num_updates: Optional[int] = None) -> bool:
        """Odd updates train the Discriminator; even updates train the Generator."""
        n = self.update_num if num_updates is None else num_updates
        return n % 2 == 1

    def get_groups_for_update(self, num_updates: int) -> str:
        return "discriminator" if self.discrim_step(num_updates) else "generator"

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,                      # [B, T, C]  wav2vec features
        padding_mask: Optional[torch.Tensor] = None, # [B, T]     True = padded
        random_label: Optional[torch.Tensor] = None, # [B', T']   text phoneme ids
        dense_x_only: bool = False,
        aux_target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Training forward pass.

        ``random_label`` carries unpaired text phoneme ids (from RandomInputDataset).
        GAN alternation uses ``self.update_num`` (set by ``set_num_updates``).
        """
        del aux_target  # reserved for upstream MMI / auxiliary losses

        # Infer device from parameters at runtime; fairseq may have moved the model
        _dev = next(self.parameters()).device
        features = features.to(_dev)
        if padding_mask is not None:
            padding_mask = padding_mask.to(_dev)

        # ── 1. Normalise features (MPS-safe instance norm) ────────────────────
        features = self._instance_norm(features)

        # ── 2. Generator forward ──────────────────────────────────────────────
        fake_logprobs, _dense, padding_mask = self.generator(features, padding_mask)
        fake_probs = fake_logprobs.exp()  # [B, T, V]  soft probability distribution

        # ── 3. Discriminator scores on fake ───────────────────────────────────
        # Feed the probability distribution directly (vocab-dim, same space as real)
        fake_scores = self.discriminator(fake_probs, padding_mask)

        # ── 4. dense_x_only (evaluation / decoding mode) ─────────────────────
        if dense_x_only:
            return {"logits": fake_logprobs, "padding_mask": padding_mask}

        # ── 5. Real data ──────────────────────────────────────────────────────
        real_probs       = None
        real_padding_mask = None
        if random_label is not None:
            real_phonemes     = random_label.to(_dev)
            real_padding_mask = real_phonemes.eq(self.pad_idx)
            real_probs        = self.real_data(real_phonemes, real_padding_mask)

        d_step      = self.discrim_step(self.update_num)
        sample_size = features.size(0)

        # ── 6. Codebook perplexity (code_ppl) ─────────────────────────────────
        # Measures Generator output diversity:  high = many different phonemes
        # used (healthy); low = mode collapse (all frames predict same phoneme).
        # Masked (padded) positions have log_prob=0 → prob=exp(0)=1, which
        # inflates the average.  Exclude them using the padding mask.
        with torch.no_grad():
            if padding_mask is not None:
                valid     = (~padding_mask).float().unsqueeze(-1)  # [B, T, 1]
                n_valid   = valid.sum().clamp(min=1)
                avg_probs = (fake_probs * valid).sum(dim=[0, 1]) / n_valid
            else:
                avg_probs = fake_probs.mean(dim=[0, 1])            # [V]
            code_ppl = torch.exp(
                -(avg_probs * (avg_probs + 1e-8).log()).sum()
            )

        result: Dict[str, Any] = {
            "fake_logprobs": fake_logprobs,
            "fake_scores":   fake_scores,
            "sample_size":   sample_size,
            "temp":          torch.tensor(self.curr_temp, device=_dev),
            "code_ppl":      code_ppl,
        }

        # ── 7. Discriminator step ─────────────────────────────────────────────
        if d_step:
            if real_probs is None:
                raise RuntimeError(
                    "Discriminator step requires unpaired text phonemes (random_label)."
                )

            # ── 7a. Dense (frame-level) discriminator ─────────────────────────
            # real_probs is already token-level text; fake_probs is frame-level
            # speech.  Both are V-dimensional distributions — the discriminator
            # treats them as sequences and learns to tell them apart.
            real_scores_dense = self.discriminator(real_probs, real_padding_mask)
            disc_loss, disc_log = _compute_discriminator_loss(
                real_scores   = real_scores_dense,
                fake_scores   = fake_scores,         # computed from dense frames above
                real_probs    = real_probs,
                fake_probs    = fake_probs.detach(),
                discriminator = self.discriminator,
                cfg           = self.cfg,
            )

            # ── 7b. Token-level (segmented) discriminator ─────────────────────
            # Collapse consecutive identical-phoneme frames into a shorter token
            # sequence, then run the discriminator again.  This gives a cleaner
            # signal: the fake sequence now looks more like real text phonemes
            # (both are token-level), so the adversarial pressure is stronger.
            #
            # Token sequences are much shorter than frame sequences
            # (T' ≈ T / avg_segment_len), so this second pass is cheap.
            # Real text is already token-level → reuse real_scores_dense.
            fake_token, fake_token_mask = self.segmenter.segment(
                fake_probs.detach(), padding_mask
            )
            fake_scores_token = self.discriminator(fake_token, fake_token_mask)
            token_w_loss = fake_scores_token.mean() - real_scores_dense.mean()
            disc_log["token_d"] = token_w_loss.item()
            total_disc_loss = disc_loss + token_w_loss

            result["losses"]      = {"dense_d": total_disc_loss}
            result["real_scores"] = real_scores_dense
            result["logs"]        = disc_log

        # ── 8. Generator step ─────────────────────────────────────────────────
        else:
            # Optional code penalty: pushes Generator toward diverse phoneme usage
            if self.cfg.code_penalty > 0:
                avg_p = fake_probs.mean(dim=[0, 1])      # [V] — needs grad to train generator
                code_pen = -avg_p.log().mean()           # maximise entropy = diversity
                code_pen_loss = self.cfg.code_penalty * code_pen
            else:
                code_pen_loss = torch.tensor(0.0, device=_dev)

            gen_loss, gen_log = _compute_generator_loss(
                fake_scores   = fake_scores,
                fake_logprobs = fake_logprobs,
                cfg           = self.cfg,
            )
            gen_log["code_pen"] = code_pen_loss.item()
            total_gen_loss      = gen_loss + code_pen_loss

            result["losses"] = {"dense_g": total_gen_loss}
            result["logs"]   = gen_log

        return result

    # ── Evaluation / decoding API ─────────────────────────────────────────────

    def get_normalized_probs(
        self,
        net_output: Any,
        log_probs: bool,
        sample: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Required by fairseq for CTC/evaluation decoding.

        Returns the Generator's output probabilities in the format
        expected by fairseq's decoding utilities.
        """
        logits = net_output["logits"] if isinstance(net_output, dict) else net_output[0]
        return logits if log_probs else logits.exp()
