#!/bin/bash

# =============================================================================
# run_setup.sh — Orchestrate the full environment setup
# =============================================================================
#
# CHANGES FROM ORIGINAL (see CHANGES.md for full rationale):
#   1. lspci GPU detection replaced with a portable macOS/Linux check.
#      On macOS: system_profiler detects Apple Silicon or discrete GPU.
#      On Linux: lspci still used for NVIDIA detection.
#   2. CUDA and GPU driver installation skipped on macOS (not supported).
#   3. install_wav2vec_u() called after install_fairseq() so wav2vec_u.py
#      is placed into the fairseq_ tree at the correct path.
#   4. All output tee'd to a timestamped log file in $INSTALL_ROOT/logs/.
#
# USAGE
#   chmod +x run_setup.sh && ./run_setup.sh
#
# =============================================================================

# ── Load all functions ────────────────────────────────────────────────────────
source "$(dirname "$0")/setup_functions.sh"

# ── Prepare log directory and start logging ───────────────────────────────────
create_dirs
LOG_FILE="$INSTALL_ROOT/logs/setup_$(date +%Y%m%d_%H%M%S).log"
# Redirect all output (stdout + stderr) to both the terminal and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

log "====================================================="
log " wav2vec-U  Setup  —  $(date)"
log " Platform  : $(uname -srm)"
log " Log file  : $LOG_FILE"
log " INSTALL_ROOT: $INSTALL_ROOT"
log "====================================================="

# =============================================================================
# 1. SYSTEM PREREQUISITES
# =============================================================================

basic_dependencies
# Installs build tools, cmake, wget, curl, boost, eigen, ffmpeg etc.
# On macOS: uses Homebrew.  On Linux: uses apt-get.

# ── GPU detection (portable macOS / Linux) ────────────────────────────────────
HAS_NVIDIA_GPU=false

if [[ "$(uname)" == "Darwin" ]]; then
    # On macOS, NVIDIA CUDA is not supported (Apple dropped NVIDIA drivers in 2018).
    # Apple Silicon Macs have the Neural Engine / MPS for GPU acceleration.
    # We still check so we can log the GPU clearly.
    if system_profiler SPDisplaysDataType 2>/dev/null | grep -qi "NVIDIA"; then
        log "[INFO] NVIDIA GPU detected on macOS — CUDA is NOT supported on macOS."
        log "       The model will use MPS (Apple Silicon) or CPU instead."
    else
        log "[INFO] macOS — using MPS (Apple Silicon GPU) or CPU. CUDA not applicable."
    fi
    # No CUDA install, no GPU driver install on macOS
else
    # Linux: check for NVIDIA GPU via lspci
    if command -v lspci &>/dev/null && lspci | grep -iq "nvidia"; then
        HAS_NVIDIA_GPU=true
        log "NVIDIA GPU detected on Linux. Proceeding with GPU setup..."
        cuda_installation
        gpu_drivers_installation
    else
        log "No NVIDIA GPU found on Linux. Using CPU-only mode."
    fi
fi

# =============================================================================
# 2. PYTHON ENVIRONMENT AND CORE FRAMEWORKS
# =============================================================================

setup_venv
# Creates an isolated Python 3.10 virtual environment using pyenv.

install_pytorch_and_other_packages
# macOS: installs PyTorch with default wheels (includes MPS support).
# Linux: installs PyTorch with CUDA 12.1 wheels if GPU is present.

install_fairseq
# Clones the Ashesi fork of fairseq and installs it in editable mode.

install_wav2vec_u
# Copies wav2vec_u.py (the custom GAN model with Generator / RealData /
# Discriminator separation of concerns) into:
#   fairseq_/examples/wav2vec/unsupervised/models/wav2vec_u.py

# =============================================================================
# 3. DOMAIN-SPECIFIC TOOLS
# =============================================================================

install_flashlight
# Installs flashlight-text and builds the flashlight-sequence C++ library.
# Required for fast beam-search decoding during evaluation.

install_kenlm
# Builds KenLM from source (C++).  Used in prepare_text to train a
# character/phoneme n-gram language model on the unlabelled text corpus.

install_rVADfast
# Clones rVADfast for voice activity detection (silence removal) from audio.

# =============================================================================
# 4. MODEL CHECKPOINTS
# =============================================================================

download_pretrained_model
# Downloads wav2vec_vox_new.pt — the pre-trained wav2vec 2.0 feature extractor.

download_languageIdentification_model
# Downloads lid.176.bin — FastText language identification model used by
# prepare_text.sh to filter out non-English text lines.

# =============================================================================
log "====================================================="
log " Setup complete!  $(date)"
log " Log saved to: $LOG_FILE"
log ""
log " Next steps:"
log "   1. Edit config.sh to set DEVICE and TEST_RUN as desired."
log "   2. Run: ./run_wav2vec.sh"
log "      (Converts audio, creates manifests, prepares features)"
log "   3. Run: ./run_gans.sh"
log "      (Trains the GAN)"
log "   4. Run: ./run_eval.sh /path/to/checkpoint.pt"
log "====================================================="
