#!/bin/bash

# =============================================================================
# config.sh — Central configuration for the wav2vec-U pipeline
# =============================================================================
#
# HOW TO USE
#   Every other script sources this file:
#       source "$(dirname "$0")/config.sh"
#
# DEVICE SWITCHING (the only line you normally need to change)
#   "auto"  → picks CUDA > MPS > CPU automatically
#   "mps"   → Apple Silicon GPU (M1/M2/M3/M4) — fast on Mac
#   "cpu"   → always works, slowest
#   "cuda"  → NVIDIA GPU (Linux / Windows only)
#
# TEST_RUN MODE
#   Set TEST_RUN=true for a tiny smoke-test (TEST_AUDIO_LIMIT / TEST_TEXT_LIMIT).
#
# FRACTIONAL "FULL" RUN (TEST_RUN=false)
#   AUDIO_DATA_PERCENT — only FLAC→WAV in convert_audio.sh (first N train/val/test
#   FLACs per split). Manifests then include every WAV you converted (no second cut).
#   TEXT_DATA_PERCENT — fraction of the LM text file in prepare_text (deterministic).
#   Use 100 for full LibriSpeech / full text.
#
# FASTER RUNS (besides less audio/text): DEVICE=mps or cuda; lower fairseq
#   max_update / larger validate_interval_updates during experiments; modest
#   dataset.num_workers on macOS (0–2 can be faster than huge values).
#
# GAN / Fairseq (train_gans) — tuned for Apple Silicon + modest dataloader load.
#   Override GAN_* below; gans_functions.sh passes them to fairseq-hydra-train.
#
# =============================================================================

# ── Compute device ──────────────────────────────────────────────────────────
# Change this single variable to switch between CPU, Apple Silicon GPU, or CUDA.
# "auto" uses GPU when available (CUDA on NVIDIA, MPS on Apple Silicon).
# GAN training reads this via gans_functions.sh → fairseq +model.device / common.cpu.
DEVICE="mps"   # auto | cpu | mps | cuda

# ── GAN training (Hydra overrides; see fairseq …/config/gan/w2vu.yaml) ─────────
# num_workers=1: avoids macOS multiprocessing churn; pairs well with MPS (single process feeds GPU).
# max_update=5000: matches the lightweight experimental setting used for comparison runs.
#   Raise to 150000 for a paper-quality long run.
# validate/save every 1000 updates: frequent enough to capture early training curves
#   without too much disk I/O.
GAN_NUM_WORKERS=6
GAN_MAX_UPDATE=5000
GAN_VALIDATE_INTERVAL_UPDATES=1000
GAN_SAVE_INTERVAL_UPDATES=1000
# Number of training utterances passed to GAN training (0 = use all).
# Set to 1892 to match the lightweight experimental baseline.
GAN_TRAIN_UTTERANCES=1892
# Log a training-stats line every N gradient updates (5 = every 5 batches).
GAN_LOG_INTERVAL=5

# ── Test-run mode ────────────────────────────────────────────────────────────
# true  = pipeline uses only a small fraction of data (fast smoke-test)
# false = pipeline uses the full dataset (production run)
TEST_RUN=false

# Number of audio files to use per split (train / val / test) in test-run mode.
# 30 files ≈ ~2 minutes of audio — enough to exercise every pipeline step
# without waiting hours.
TEST_AUDIO_LIMIT=30

# Number of text lines from the language-model corpus used in test-run mode.
# 200 lines is enough for KenLM to build a tiny n-gram LM and for
# prepare_text.sh to produce the required phoneme files.
TEST_TEXT_LIMIT=200

# Percent of LibriSpeech FLACs to convert when TEST_RUN=false (see header).
# 100 = convert all FLACs under data/corpora/LibriSpeech. Ignored when TEST_RUN=true.
# run_wav2vec.sh re-runs conversion when train WAV count is below what this percent implies.
AUDIO_DATA_PERCENT=20
TEXT_DATA_PERCENT=65

# ── Dataset name ─────────────────────────────────────────────────────────────
# Passed to prepare_audio / prepare_text to name output subdirectories.
DATASET_NAME="librispeech"

# ── Python version ───────────────────────────────────────────────────────────
PYTHON_VERSION="3.10"

# ── Export all variables so child processes can read them ────────────────────
export DEVICE
export TEST_RUN
export TEST_AUDIO_LIMIT
export TEST_TEXT_LIMIT
export AUDIO_DATA_PERCENT
export TEXT_DATA_PERCENT
export DATASET_NAME
export PYTHON_VERSION
export GAN_NUM_WORKERS
export GAN_MAX_UPDATE
export GAN_VALIDATE_INTERVAL_UPDATES
export GAN_SAVE_INTERVAL_UPDATES
export GAN_TRAIN_UTTERANCES
export GAN_LOG_INTERVAL
