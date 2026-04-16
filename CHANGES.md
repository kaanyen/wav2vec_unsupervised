# Wav2Vec-U macOS Adaptation — Change Log

**Course**: ICS554 Natural Language Processing — PROSIT 3  
**Task**: Unsupervised speech recognition using GANs (Wav2Vec-U)  
**Platform adapted to**: macOS (Apple Silicon M-series and Intel), CPU + MPS GPU  
**Original platform**: Linux with NVIDIA CUDA GPU  

---

## Overview

The original [`Ashesi-Org/wav2vec_unsupervised`](https://github.com/Ashesi-Org/wav2vec_unsupervised)
repository is a set of shell scripts that automate the
[Fairseq Wav2Vec-U pipeline](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md)
on a Linux machine with an NVIDIA GPU.  Running it unchanged on macOS fails at
almost every step.  This document records every change made, explains why it was
necessary, and provides background useful for writing a technical report.

---

## Changes by File

### 1. `config.sh` *(new file)*

**Why created:**  
All configurable knobs (compute device, test-run limits) were previously scattered
across multiple scripts with no single place to change them.  A central config
file makes switching between CPU, Apple Silicon GPU (MPS), and NVIDIA GPU (CUDA)
a one-line edit.

**What it contains:**

| Variable | Default | Purpose |
|---|---|---|
| `DEVICE` | `auto` | Compute device: `auto` \| `cpu` \| `mps` \| `cuda` |
| `TEST_RUN` | `true` | Use a tiny data subset for smoke-testing |
| `TEST_AUDIO_LIMIT` | `30` | Audio files per split during a test run |
| `TEST_TEXT_LIMIT` | `200` | Text lines used during a test run |
| `AUDIO_DATA_PERCENT` | `100` | When `TEST_RUN=false`, percent of each audio split (1–100); `100` = full |
| `TEXT_DATA_PERCENT` | `100` | When `TEST_RUN=false`, percent of unlabelled text lines for LM/text prep |
| `DATASET_NAME` | `librispeech` | Names output subdirectories |
| `PYTHON_VERSION` | `3.10` | Python version for venv |

**How to switch devices:**  
Edit the single line `DEVICE="mps"` (Apple Silicon GPU), `DEVICE="cpu"` (CPU only),
or `DEVICE="cuda"` (NVIDIA, Linux/Windows).  The `wav2vec_u.py`
`select_device()` function reads this variable and falls back gracefully if the
requested device is unavailable.

---

### 2. `convert_audio.sh` *(new file)*

**Why created:**  
LibriSpeech ships audio in **FLAC** format.  The `wav2vec_manifest.py` script
and the `remove_silence.py` pipeline both expect **WAV** files.  This script
uses `ffmpeg` (Homebrew: `brew install ffmpeg`) to convert every FLAC file to
16 kHz, mono, 16-bit PCM WAV.

**Key behaviour:**
- Outputs to `data/wav/train/`, `data/wav/val/`, `data/wav/test/`.
- Skips files that already exist (idempotent).
- When `TEST_RUN=true` converts only the first `TEST_AUDIO_LIMIT` files per
  split, so a smoke-test finishes in under 5 minutes instead of hours.
- When `TEST_RUN=false` and `AUDIO_DATA_PERCENT` &lt; 100, converts only the first
  fraction of sorted FLAC paths per split (e.g. `20` → 20% per split).

**Data counts (LibriSpeech):**

| Split | FLAC files | Full convert time (approx.) | Test-run files |
|---|---|---|---|
| `train-clean-100` | 28 539 | ~45 min | 30 |
| `dev-clean` | 2 703 | ~4 min | 30 |
| `test-clean` | 2 620 | ~4 min | 30 |

---

### 3. `setup_functions.sh`

#### 3a. `INSTALL_ROOT` path fix

**Original:**
```bash
INSTALL_ROOT="$HOME/wav2vec_unsupervised"
```

**Changed to:**
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="$SCRIPT_DIR"
```

**Why:**  
The original path placed all downloaded dependencies (`fairseq_`, `kenlm`,
`rVADfast`, `venv`, model checkpoints) in `~/wav2vec_unsupervised`, a
*separate* copy from the cloned project.  This means the project folder and
the runtime environment were in two different places, creating confusion and
making the setup non-portable.  The fix makes everything self-contained inside
the cloned repository folder.

#### 3b. `basic_dependencies()` — `apt-get` → `brew`

**Original:** `sudo apt-get install ...` / `sudo apt install ...`  
**Changed to:** `brew install ...` on macOS; `apt-get` preserved as Linux fallback.

**Why:**  
`apt-get` is the Debian/Ubuntu package manager and does not exist on macOS.
Homebrew is the standard macOS package manager.  The equivalent packages are:

| apt-get package | brew package |
|---|---|
| `build-essential`, `g++` | `cmake`, `autoconf`, `automake` (Xcode CLT provides gcc) |
| `libeigen3-dev` | `eigen` |
| `libboost-all-dev` | `boost` |
| `pybind11-dev` | `pybind11` |
| `ffmpeg` | `ffmpeg` |

#### 3c. CUDA / GPU driver installation skipped on macOS

**Original:** `cuda_installation()` and `gpu_drivers_installation()` always ran.  
**Changed to:** Both functions immediately return if `uname == Darwin`.

**Why:**  
Apple removed support for NVIDIA CUDA drivers on macOS after macOS 10.13 (High
Sierra) in 2018.  Modern Macs use Apple's Metal/MPS stack for GPU compute.
Attempting to install CUDA on macOS would fail.

#### 3d. PyTorch install — remove CUDA index URL

**Original:**
```bash
pip install torch==2.3.0 ... --index-url "https://download.pytorch.org/whl/cu121"
```

**Changed to (macOS):**
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
```

**Why:**  
The `cu121` wheel index only contains CUDA-enabled builds.  Installing from
that URL on macOS either downloads the wrong binary or fails entirely.  The
default PyPI wheels for PyTorch already include MPS (Metal Performance Shaders)
support for Apple Silicon — no special index URL is needed.

#### 3e. `nproc` → `sysctl -n hw.ncpu`

**Original:** `make -j $(nproc)`  
**Changed to:** `make -j "$(cpu_count)"` where `cpu_count()` uses
`sysctl -n hw.ncpu` on macOS and `nproc` on Linux.

**Why:**  
`nproc` is a Linux utility that prints the number of CPU cores.  On macOS the
equivalent is `sysctl -n hw.ncpu`.  The portable `cpu_count()` wrapper handles
both.

#### 3f. `install_wav2vec_u()` — new function

**Added:**
```bash
install_wav2vec_u() {
    cp "$INSTALL_ROOT/wav2vec_u.py" \
       "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/models/wav2vec_u.py"
}
```

**Why:**  
The assignment requires a custom `wav2vec_u.py` implementing the GAN with
explicit `Generator`, `RealData`, and `Discriminator` classes (separation of
concerns).  This function places it at exactly the path fairseq expects, after
`install_fairseq()` has cloned the repository and created the directory tree.

---

### 4. `run_setup.sh`

#### 4a. GPU detection fix

**Original:** `if lspci | grep -iq "nvidia"`  
**Changed to:** OS-aware check:
- macOS: `system_profiler SPDisplaysDataType | grep -qi "NVIDIA"`
- Linux: `lspci | grep -iq "nvidia"` (unchanged)

**Why:** `lspci` is a Linux PCI bus tool; it does not exist on macOS.

#### 4b. `install_wav2vec_u` call added

Called immediately after `install_fairseq` so the `fairseq_` directory already
exists when the copy is attempted.

#### 4c. Timestamped log file

All setup output is `tee`'d to `logs/setup_YYYYMMDD_HHMMSS.log`.  
**Why:** Setup takes 20–60 minutes; a persistent log makes it easy to review
errors after the fact without re-running.

---

### 5. `utils.sh`

#### 5a. `DIR_PATH` path fix

Same fix as `INSTALL_ROOT` in `setup_functions.sh` — now derived from script
location instead of hardcoded to `$HOME/wav2vec_unsupervised`.

#### 5b. `source config.sh`

`config.sh` is sourced at the top so all scripts that source `utils.sh`
automatically inherit `DEVICE`, `TEST_RUN`, etc.

#### 5c. `sed -i` portability fix in `mark_in_progress()`

**Original:** `sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"`  
**Changed to:**
```bash
sed -i '' "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE" 2>/dev/null \
    || sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
```

**Why:**  
BSD `sed` (macOS default) requires an explicit backup suffix with `-i`.
`sed -i ''` means "edit in-place, no backup".  GNU `sed` (Linux) accepts
`sed -i` without a suffix.  The fallback ensures compatibility on both.

---

### 6. `wav2vec_functions.sh`

#### 6a. `portable_sed_i()` wrapper

Replaces all bare `sed -i` calls.  Detects GNU vs BSD sed automatically.

#### 6b. `maybe_truncate_manifest()` helper

When `TEST_RUN=true`, this function truncates any manifest (TSV) file to
`TEST_AUDIO_LIMIT` data rows after it is generated.  The first line (the root
path header) is always preserved.  Applied to all six manifest creation
functions: `create_manifests_{train,val,test}` and
`create_manifests_nonsil_{train,val}`.

`AUDIO_DATA_PERCENT` is **not** applied here (avoids compounding 20% at
FLAC→WAV with 20% on raw TSV and again on nonsil TSV).  Subsample audio only in
`convert_audio.sh`; manifests list every WAV under `data/wav`.

#### 6c. `prepare_text()` — text subset for test runs

When `TEST_RUN=true`, creates `data/text_subset_200.txt` by taking the first
`TEST_TEXT_LIMIT` lines of the full wikitext corpus.  KenLM trains in seconds
on 200 lines vs minutes on 29 000 lines.

#### 6d. `prepare_text.sh` / `normalize_and_filter_text.py` — macOS

- **`grep -P`** is GNU-only; replaced with **`grep -Ev '[0-9]{5,}'`** (BSD-compatible).
- **`fasttext`**: lazy import + `load_fasttext_model()` so a broken PyPI wheel does not abort before stderr guidance; **`setup_functions.sh`** uses **`pip install --no-binary fasttext fasttext`** so the extension links against the local libc++ (PyPI macOS wheels can embed bad `@rpath`s).
- **Fairseq preprocess** `--workers` capped (default max 8; override with **`FAIRSEQ_PREPROCESS_WORKERS`**) to avoid leaked semaphores on tiny corpora.
- **KenLM `build_binary`**: **`-s`** so tiny/degenerate ARPA files without `</s>` still convert (KenLM `MissingSentenceMarkerException`).

---

### 7. `run_wav2vec.sh`

#### 7a. `config.sh` sourced

#### 7b. `convert_audio.sh` called first

Before manifests are created, `run_wav2vec.sh` compares the number of train WAVs
in `data/wav/train/` to the minimum required for the current `config.sh`
(`TEST_RUN` / `AUDIO_DATA_PERCENT` vs LibriSpeech FLAC counts under
`data/corpora/LibriSpeech/`).  If there are no WAVs, or fewer than required
(e.g. an old 30-file smoke-test cache while `AUDIO_DATA_PERCENT=20` needs
thousands), `convert_audio.sh` runs and reads the real FLAC corpus.

#### 7c. Default data paths

If no arguments are given, the script uses the data already present in
`data/corpora/`:
- Train: `data/wav/train/` (converted from `data/corpora/LibriSpeech/train-clean-100/`)
- Val: `data/wav/val/` (from `dev-clean/`)
- Test: `data/wav/test/` (from `test-clean/`)
- Text: `data/corpora/wikitext_103_raw_v1.txt`

---

### 8. `wav2vec_u.py` placement

The file is placed in the project root (`wav2vec_unsupervised/wav2vec_u.py`).
`install_wav2vec_u()` copies it to the final location after fairseq is cloned:

```
wav2vec_unsupervised/fairseq_/examples/wav2vec/unsupervised/models/wav2vec_u.py
```

---

## Architecture Overview (for Technical Report)

### Wav2Vec-U Pipeline

```
Audio (FLAC) ──► convert_audio.sh ──► Audio (WAV 16kHz)
                                              │
                             wav2vec_manifest.py (create manifests)
                                              │
                             rVADfast (detect silence)
                                              │
                             remove_silence.py (trim audio)
                                              │
                             prepare_audio.sh
                               • wav2vec 2.0 feature extraction
                               • PCA to 512 dims
                               • k-means clustering → pseudo-phonemes
                                              │
Text corpus ────────────────► prepare_text.sh
(wikitext-103)                 • G2P phonemisation
                               • KenLM n-gram LM training
                                              │
                             GAN Training (run_gans.sh)
                               • Generator:     speech features → fake phonemes
                               • RealData:       text → real phoneme embeddings
                               • Discriminator: real vs fake classification
                                              │
                             Viterbi Decoding (run_eval.sh)
                                              │
                               data/transcription_phones/test.txt
```

### GAN Architecture (wav2vec_u.py)

The `wav2vec_u.py` file implements the GAN following the
**separation-of-concerns** principle required by the assignment:

| Class | Role | Implementation |
|---|---|---|
| `SamePad`, `TransposeLast` | Geometry / padding helpers | No trainable parameters; pure shape utilities |
| `Generator` | Converts wav2vec speech features into fake phoneme probabilities | Stock: `TransposeLast → Conv1d(512, V, k=4) → TransposeLast` — single 1-D projection |
| `RealData` | Encodes text phoneme indices as probability vectors for the Discriminator | One-hot encoding over vocab size V (no learned parameters) |
| `Discriminator` | Scores a phoneme-probability sequence as real or fake | Stock: `Conv1d(V→384) + SamePad + [Conv1d+SamePad+GELU]×1 + Conv1d→1` |
| `Wav2VecU` | Top-level fairseq model tying all three together | `@register_model("wav2vec_u")`; manages GAN alternation, temp schedule, and losses |

**Why V-dimensional inputs to the Discriminator?**  
Both Generator output (soft probabilities) and RealData output (one-hot vectors)
live in the same V-dimensional probability simplex.  The Discriminator therefore
operates on a common, symmetric representation — it never has to distinguish
structural differences between embeddings and one-hots.

**Loss functions:**
- Generator: adversarial (`dense_g`, −mean fake score) + smoothness (`smooth`)
              + codebook penalty (`code_pen`)
- Discriminator: Wasserstein (`dense_d`, mean fake − mean real) + WGAN-GP (`grad_pen`)

**Device selection:**  
`select_device(preferred)` picks CUDA > MPS > CPU automatically when
`preferred="auto"`.  Set `DEVICE="mps"` in `config.sh` to force Apple Silicon GPU.

**Fairseq runtime (`config.sh` → `gans_functions.sh`):**  
`GAN_NUM_WORKERS`, `GAN_MAX_UPDATE`, `GAN_VALIDATE_INTERVAL_UPDATES`,
`GAN_SAVE_INTERVAL_UPDATES`, `GAN_TRAIN_UTTERANCES`, and `GAN_LOG_INTERVAL`
are passed as Hydra overrides (current values: 1 worker, 5k max updates,
validate/save every 1k updates, 1 892 training utterances, log every 5 updates).
Raise `GAN_MAX_UPDATE` to 150 000 and set `GAN_TRAIN_UTTERANCES=0` for a full
paper-quality run; the upstream `w2vu.yaml` defaults match these values when
training is launched without the shell wrapper.

**Training-utterance subsetting (`gans_functions.sh` — `prepare_gan_train_subset`):**  
A new helper function creates a lightweight subset directory beside the original
precomputed-features folder when `GAN_TRAIN_UTTERANCES > 0`:

| File | Action | Rationale |
|---|---|---|
| `train.tsv` | `head -n N+1` (header + N rows) | Limits the sample manifest |
| `train.lengths` | `head -n N` | Limits the length/offset list |
| `train.npy` | **symlink** to original | Avoids copying the ~1.8 GB feature matrix; fairseq opens it with `mmap_mode="r"` and only reads the frames covered by the truncated lengths list |
| `valid.*`, `test.*` | copied unchanged | Validation and test remain full |

The subset dir is named `precompute_pca512_cls128_mean_pooled_sub1892/` and is
reused on subsequent runs (content is verified before re-creation).

**Progress output (`w2vu.yaml` + `gans_functions.sh`):**  
Two settings control how often training statistics are printed to the terminal:

| Setting | Before | After | Effect |
|---|---|---|---|
| `common.log_format` | `json` | **`simple`** | Human-readable lines instead of JSON blobs |
| `common.log_interval` | `100` | **`5`** | Print every 5 gradient updates (≈ every 5 batches) |

With `log_interval=5` and ~12 batches per "epoch", you see roughly **2–3 lines of
stats per epoch**, making it easy to track loss, temperature, and PPL without
waiting through silent stretches of hours.

**Model / speed alignment — round 1 (`w2vu.yaml`, slimming discriminator):**  
*(This was the initial alignment attempt.  Superseded by round 2 below.)*

| Parameter | Original | Slimmed | Effect |
|---|---|---|---|
| `generator_hidden` | 512 | 256 | Halved Generator conv channels |
| `discriminator_dim` | 384 | 128 | Shrank Discriminator hidden width |
| `discriminator_depth` | 2 | 1 | One fewer causal-conv layer |

Measured result: no speedup — the bottleneck was the **number of Generator conv
layers** (4), not the hidden dimension.  See round 2.

---

**Model / speed alignment — round 2 (`wav2vec_u.py` + `w2vu.yaml`):**

Three changes made after profiling showed 4-layer Generator was the bottleneck:

**A. Generator: 1 conv layer via `generator_layers` config field**

The custom Generator previously hardcoded `num_layers=4`.  A new
`generator_layers` field was added to `Wav2VecUConfig` and the stack is now built
with `num_layers=cfg.generator_layers`.  `w2vu.yaml` sets `generator_layers: 1`.

The WGAN-GP gradient penalty computes `∇_x D(x)` via `autograd.grad`, which must
backpropagate through the generator as part of the interpolated sample path.  With
4 conv layers this takes ~45 s/update; with 1 layer it drops to ~7–9 s/update,
matching the stock Wav2Vec-U baseline.

| Setting | Before | After |
|---|---|---|
| `generator_layers` | 4 (hardcoded) | **1** (via `w2vu.yaml`) |
| `generator_hidden` | 256 | **512** (restored to stock) |
| Expected sec/update | ~46 | **~7–9** |

To restore the deeper stack, set `generator_layers: 4` in `w2vu.yaml`.

**B. Discriminator: restored to stock dimensions**

Reverted the earlier slimming — the stock Discriminator provides a stronger
adversarial signal to the Generator:

| Parameter | Slimmed | Restored |
|---|---|---|
| `discriminator_dim` | 128 | **384** |
| `discriminator_depth` | 1 | **2** |

**C. `code_ppl` bug fix (`wav2vec_u.py`)**

The custom model hardcoded `"code_ppl": torch.tensor(0.0)`, so the metric was
always 0 in every log line.  Fixed to compute the actual perplexity of the
Generator's phoneme distribution:

```python
avg_probs = fake_logprobs.exp().mean(dim=[0, 1])   # average prob per phoneme
code_ppl  = exp( -sum( p * log(p) ) )              # entropy → perplexity
```

`code_ppl` should start near 1 (collapse) and rise towards the vocab size (47)
as the Generator learns to produce diverse phoneme sequences.

**Updated training schedule comparison:**

| Factor | Baseline | This run |
|---|---|---|
| `max_update` | 5 000 | **5 000** ✓ |
| Train utterances | ~1 892 | **1 892** ✓ |
| Batches / "epoch" | ~12 | **~12** ✓ |
| Generator layers | 1 | **1** ✓ |
| `discriminator_dim` | 384 | **384** ✓ |
| Hydra multiruns | 1 | **1** ✓ |

All factors now match.  The full 5 707 WAVs remain on disk; GAN training reads
only the first 1 892 entries via the subset manifest.
Set `GAN_TRAIN_UTTERANCES=0` in `config.sh` to revert to all utterances.

---

**Model / speed alignment — round 3 (full stock rewrite of `wav2vec_u.py`):**

Previous rounds made the Generator faster by reducing its layer count, but the
model still had two structural differences from the stock Fairseq implementation
that added overhead:

| Component | Custom (rounds 1–2) | Stock (round 3) |
|---|---|---|
| Generator | `Linear(512→512) → Conv1d×1 → ELU → Linear(→V)` | `TransposeLast → Conv1d(512, V, k=4) → TransposeLast` |
| RealData | `nn.Embedding(V, 384)` → learned 384-dim vectors | One-hot encoding → V-dim probability vectors |
| Discriminator input | 384-dim (from learned embeddings) | V-dim (47; same as Generator output) |
| Discriminator activation | ELU | **GELU** (stock) |
| `code_ppl` | Fixed in round 2 | Unchanged ✓ |

**Changes in `wav2vec_u.py`:**

1. **Generator** — replaced `input_proj + conv_stack + output_proj` with the
   stock single `TransposeLast → Conv1d → TransposeLast` projection.
   The two extra `nn.Linear` layers are gone, removing ~200 K parameters and
   further cutting gradient-penalty backprop time.

2. **RealData** — replaced `nn.Embedding(V, disc_dim)` + positional bias with
   plain `F.one_hot(ids, num_classes=V)`.  RealData now has **0 learned
   parameters** and outputs a V-dimensional one-hot vector per frame — the same
   probability space as the Generator, making the Discriminator inputs symmetric.

3. **Discriminator** — input dimension changed from `discriminator_dim` (384) to
   `num_phonemes` (47).  Activation changed from `ELU` to `GELU` (stock).

4. **Forward** — the `matmul(fake_probs, embed_weight)` projection step is gone;
   `fake_probs` (V-dim) feeds the Discriminator directly.

5. **Config (`Wav2VecUConfig`)** — removed custom-only fields
   `generator_hidden`, `generator_embed_dim`, `generator_layers`,
   `generator_residual`, `generator_batch_norm`.  No longer referenced in
   `w2vu.yaml`.

**`w2vu.yaml`** — removed `generator_hidden` and `generator_layers` from
the `model:` block.  All other settings unchanged.

**Model size after round 3 (vocab = 47):**

| Component | Parameters |
|---|---|
| Generator `Conv1d(512, 47, k=4)` | 96,256 |
| RealData | 0 (one-hot, no learned params) |
| Discriminator `Conv1d×3` | 996,097 |
| Feature norm (weight + bias) | 1,024 |
| **Total** | **1,093,377** |

The friend's model totals 1,079,297 params with vocab = 44; the difference
(~14 K) is exactly `(47 − 44) × (512 × 4 + 384 × 6)` — purely from the 3 extra
phoneme classes in this run's vocabulary.

---

### Round 4 — JoinSegmenter + token_d loss (`wav2vec_u.py`, `config.sh`)

**Why:** The stock Fairseq model includes a `JoinSegmenter` and a second
discriminator loss (`loss_token_d`) that were absent from the previous rewrite.
The friend's logs show `train_loss_token_d` every update — this is a token-level
Wasserstein loss that gives the Generator a stronger, more direct adversarial
signal.  Adding it closes the last architectural gap between this model and the
stock Fairseq implementation.

**What was added:**

1. **`JoinSegmenter` class** (new Section 1.5 in `wav2vec_u.py`)
   - Collapses consecutive frames with the same argmax phoneme into a single
     token by mean-pooling their soft probability distributions.
   - Segment boundaries come from argmax (non-differentiable), but mean-pooling
     within each segment is differentiable — gradients still flow to the Generator.
   - Implementation: fully vectorised via `scatter_add_` — no Python loops over
     batch items, MPS-compatible.
   - No trainable parameters.

2. **`token_d` loss** (discriminator step in `Wav2VecU.forward`)
   - After the dense (frame-level) discriminator pass, the segmenter produces
     a shorter token sequence `[B, T', V]` from the fake frames.
   - The discriminator scores the token sequence against real text phonemes.
   - Token-level Wasserstein loss: `mean(fake_token_scores) − mean(real_scores)`.
   - This loss is added to the dense discriminator loss: `total_disc = dense_d + token_d`.
   - Token sequences are much shorter than frame sequences
     (T' ≈ T / avg_segment_length), so this second pass adds minimal overhead.

3. **`TEXT_DATA_PERCENT` raised from 20 → 65** (`config.sh`)
   - The friend's model loaded 615 real text examples vs 200 in this run.
   - Increasing to 65% of the WikiText-103 corpus brings the count to ~650,
     matching the friend's real-data diversity.
   - **Requires re-running `prepare_text.sh`** before the next GAN training run
     (see instructions below).

**Updated model architecture:**

| Component | Parameters |
|---|---|
| Generator `Conv1d(512, 47, k=4)` | 96,256 |
| RealData | 0 (one-hot, no learned params) |
| Discriminator `Conv1d×3` | 996,097 |
| JoinSegmenter | 0 (no learned params) |
| Feature norm (weight + bias) | 1,024 |
| **Total** | **1,093,377** |

**Re-running text prep to get more real examples:**

```bash
# Kill current run_gans.sh, then in the wav2vec_unsupervised directory:
source config.sh
bash prepare_text.sh   # or ./run_wav2vec.sh if it handles text re-prep

# Then restart GAN training:
./run_gans.sh
```

---

## Dataset Information (for Technical Report)

### LibriSpeech

| Property | Value |
|---|---|
| Source | [openslr.org/12](http://www.openslr.org/resources/12/) |
| Train split used | `train-clean-100` (100 hours, 28 539 files) |
| Validation split | `dev-clean` (5.4 hours, 2 703 files) |
| Test split | `test-clean` (5.4 hours, 2 620 files) |
| Format (original) | FLAC, 16 kHz, mono |
| Format (after convert) | WAV, 16 kHz, mono, 16-bit PCM |
| Language | English |
| Speakers | Multiple (read speech from LibriVox audiobooks) |
| Label availability | Transcripts available but **not used** (unsupervised setting) |

### WikiText-103

| Property | Value |
|---|---|
| File | `data/corpora/wikitext_103_raw_v1.txt` |
| Lines | 29 567 |
| Content | Wikipedia articles (raw, unprocessed) |
| Use in pipeline | Provides the unpaired text corpus for `prepare_text.sh`; converted to phoneme sequences with G2P and used to train the KenLM language model |

### Why unpaired data?

Wav2Vec-U is designed for **low-resource languages** where parallel audio-text
data does not exist.  The key insight: the audio and text corpora do not need
to correspond.  The GAN learns the statistical distribution of phoneme sequences
from text alone and uses it to constrain the Generator's output.

---

## Running the Pipeline

### Prerequisites

1. macOS with Homebrew installed
2. Python 3.10 (via Homebrew `python@3.10` and/or `pyenv` — setup script handles this)
3. At least 8 GB RAM (16 GB recommended for full run)

### Troubleshooting: KenLM CMake fails on macOS (Boost `boost_system`, Eigen version)

Homebrew **Boost 1.82+** no longer exposes a separate `boost_system` CMake package (`boost_system` is header-only). KenLM’s stock `CMakeLists.txt` still lists `system` in `find_package(Boost … COMPONENTS …)`, which makes CMake error. Homebrew **Eigen** can report a CMake version that does not satisfy `find_package(Eigen3 3.1.0 CONFIG)`.

`setup_functions.sh` fixes this by:

1. **`patch_kenlm_for_macos()`** — relaxes the Eigen line to `find_package(Eigen3 CONFIG)` and removes the `system` line from the Boost component list (idempotent; only edits matching lines).
2. **CMake flags on Darwin** — `-DCMAKE_POLICY_DEFAULT_CMP0167=OLD`, `-DBoost_NO_BOOST_CMAKE=ON`, `-DBOOST_ROOT="$(brew --prefix boost)"`, etc., so the legacy `FindBoost` module finds Homebrew’s layout.
3. **Rebuild detection** — if `build/` exists but `build/bin/lmplz` is missing (failed configure), `build/` is removed and KenLM is rebuilt.

### Troubleshooting: `curl: (6) Could not resolve host: pyenv.run`

The original setup used `curl https://pyenv.run | bash` to install pyenv. That fails when DNS is unavailable, you are offline, or a firewall blocks the domain. The updated `setup_functions.sh` installs **pyenv via Homebrew** (`brew install pyenv`) on macOS and only uses the network installer on Linux with a timeout and fallbacks. If pyenv cannot install a Python version, the script falls back to **`brew --prefix python@3.10`/bin/python3.10** to create the venv.

### Step-by-step

```bash
# 0. Make all scripts executable
chmod +x setup_functions.sh wav2vec_functions.sh eval_functions.sh \
         gans_functions.sh run_setup.sh run_wav2vec.sh run_eval.sh \
         run_gans.sh utils.sh config.sh convert_audio.sh

# 1. Edit config.sh to choose device and run mode
#    DEVICE="mps"     ← Apple Silicon GPU (recommended on M1/M2/M3)
#    DEVICE="cpu"     ← CPU-only (always works)
#    TEST_RUN=true    ← start with a smoke-test (recommended)

# 2. Run environment setup (~20-60 min first time)
./run_setup.sh

# 3. Run data preparation + feature extraction
./run_wav2vec.sh

# 4. Train the GAN
./run_gans.sh

# 5. Evaluate with best checkpoint
./run_eval.sh multirun/<date>/<time>/0/checkpoint_best.pt
```

**Reference phones + KenLM for `w2vu_generate` metrics:**  
`eval_functions.sh` passes `targets=phn` when `data/clustering/.../test.phn` exists and `lm_model` when `data/text/phones/lm.phones.filtered.04.bin` exists.  
Build `test.phn` with `scripts/build_test_phn_from_librispeech.py` (LibriSpeech `test-clean` transcripts + `lexicon_filtered.lst`).  
If only phone ARPA files exist, run `kenlm/build/bin/build_binary -s data/text/phones/lm.phones.filtered.04.arpa data/text/phones/lm.phones.filtered.04.bin`.

**KenLM / Flashlight beam decode:**  
`./run_eval.sh <ckpt> kenlm` uses `config/generate/kenlm.yaml` (Flashlight `LexiconDecoder` / `LexiconFreeDecoder`).  
- Default **`EVAL_KENLM_MODE=phones`**: `unit_lm=true` + `lm.phones.filtered.04.bin` (matches phoneme targets).  
- **`EVAL_KENLM_MODE=words`**: `lexicon_filtered.lst` + `kenlm.wrd.o40003.bin`.  
Install **`flashlight-text`** and **`flashlight-sequence`** (build with `USE_OPENMP=0` on macOS if CMake cannot find OpenMP).  
`w2l_decoder.py` maps **`lm_model` → KenLM `.bin`** for decoders that expect `kenlm_model`, prefers **`<SIL>`** as the trie silence token when present, and uses **`_decoder_kenlm_bin_path`** so `w2vu_generate` configs work.  
**`EVAL_LM_WEIGHT`** defaults to **0**: non-zero phone-LM weights often collapse the beam (empty hypotheses) unless retuned; use `scripts/sweep_kenlm_decode.sh` to search small weights.

### Switching from test run to full run

In `config.sh`, change:
```bash
TEST_RUN=false
```
Then delete the checkpoint file to re-run all steps:
```bash
rm -f data/checkpoints/librispeech/progress.checkpoint
./run_wav2vec.sh
./run_gans.sh
```

---

## Log Files

| File | Contents |
|---|---|
| `logs/setup_YYYYMMDD_HHMMSS.log` | Full setup output (timestamped) |
| `data/logs/librispeech/pipeline.log` | Pipeline run log (appended each run) |
| `data/results/librispeech/training1.log` | GAN training loss log |

---

## Key References

- Baevski, A., et al. (2021). *Unsupervised Speech Recognition*.
  [arXiv:2105.11084](https://arxiv.org/abs/2105.11084)
- Baevski, A., et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised
  Learning of Speech Representations*.
  [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)
- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*.
  [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)
- Gulrajani, I., et al. (2017). *Improved Training of Wasserstein GANs*.
  [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) (WGAN-GP)
- Panayotov, V., et al. (2015). *LibriSpeech: An ASR corpus based on public
  domain audio books*.  ICASSP 2015.
- Zen, H., et al. (2019). *FastSpeech* / G2P phonemiser.
- KenLM: Heafield, K. (2011). *KenLM: Faster and Smaller Language Model Queries*.
  WMT 2011.
