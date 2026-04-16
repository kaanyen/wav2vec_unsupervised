#!/bin/bash

# =============================================================================
# utils.sh — Shared variables, paths, and helper functions
# =============================================================================
#
# CHANGES FROM ORIGINAL (see CHANGES.md for full rationale):
#   1. DIR_PATH now derived from script location (was hardcoded to
#      $HOME/wav2vec_unsupervised, which placed outputs outside the project).
#   2. config.sh is sourced here so DEVICE, TEST_RUN etc. are available to
#      every script that sources utils.sh.
#   3. sed -i '' used instead of sed -i for macOS compatibility.
#      (GNU sed accepts "sed -i 's/...'" but BSD sed on macOS requires
#       an explicit backup extension: "sed -i '' 's/...'")
#   4. LOG_DIR created before first use so tee -a never fails.
#
# =============================================================================

# ── Derive project root from script location ──────────────────────────────────
# Works regardless of cwd when the script is sourced.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source central config (DEVICE, TEST_RUN, TEST_AUDIO_LIMIT, …) ─────────────
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    source "$SCRIPT_DIR/config.sh"
fi

# ==================== PATHS ====================

DIR_PATH="$SCRIPT_DIR"                              # project root
DATA_ROOT="$DIR_PATH/data"                          # all generated data lives here
FAIRSEQ_ROOT="$DIR_PATH/fairseq_"                   # cloned fairseq repository
KENLM_ROOT="$DIR_PATH/kenlm/build/bin"              # KenLM binaries
VENV_PATH="$DIR_PATH/venv"                          # Python virtual environment
RVAD_ROOT="$DIR_PATH/rVADfast/src/rVADfast"         # rVADfast source

GANS_OUTPUT_PHONES="$DATA_ROOT/transcription_phones"


# ── fairseq source files that need small patches ─────────────────────────────
SPEECHPROCS="$DIR_PATH/rVADfast/src/rVADfast/speechproc/speechproc.py"
PREPARE_AUDIO="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh"
ADD_SELF_LOOP_SIMPLE="$FAIRSEQ_ROOT/examples/speech_recognition/kaldi/add-self-loop-simple.cc"
OPENFST_PATH="$DIR_PATH/fairseq/examples/speech_recognition/kaldi/kaldi_initializer.py"


# ==================== PIPELINE ARGUMENTS ====================

NEW_SAMPLE_PCT=0.5      # fraction of audio used for k-means clustering
MIN_PHONES=3            # minimum phonemes per word
NEW_BATCH_SIZE=32       # batch size for audio feature extraction
PHONEMIZER="G2P"        # phonemiser (G2P = grapheme-to-phoneme)
LANG="en"               # language code


# ── Model checkpoints ────────────────────────────────────────────────────────
FASTTEXT_LIB_MODEL="$DIR_PATH/lid_model/lid.176.bin"
MODEL="$DIR_PATH/pre-trained/wav2vec_vox_new.pt"


# ── Dataset name ─────────────────────────────────────────────────────────────
DATASET_NAME="${DATASET_NAME:-librispeech}"


# ── Output directories (created by create_dirs) ──────────────────────────────
MANIFEST_DIR="$DATA_ROOT/manifests"
NONSIL_AUDIO="$DATA_ROOT/processed_audio/"
MANIFEST_NONSIL_DIR="$DATA_ROOT/manifests_nonsil"
CLUSTERING_DIR="$DATA_ROOT/clustering/$DATASET_NAME"
RESULTS_DIR="$DATA_ROOT/results/$DATASET_NAME"
CHECKPOINT_DIR="$DATA_ROOT/checkpoints/$DATASET_NAME"
LOG_DIR="$DATA_ROOT/logs/$DATASET_NAME"
TEXT_OUTPUT="$DATA_ROOT/text"

# Pipeline progress checkpoint (tracks which steps have been completed)
CHECKPOINT_FILE="$CHECKPOINT_DIR/progress.checkpoint"


# ==================== HELPER FUNCTIONS ====================

# Log with timestamp; writes to both stdout and the pipeline log file.
log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    local line="[$timestamp] $message"
    # Print to terminal
    echo "$line"
    # Append to log file if the directory already exists
    if [ -d "$LOG_DIR" ]; then
        echo "$line" >> "$LOG_DIR/pipeline.log"
    fi
}

# Check if a pipeline step has already been completed (idempotent runs).
is_completed() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:COMPLETED$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Check if a step is marked in-progress (useful for crash recovery).
is_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        grep -q "^$step:IN_PROGRESS$" "$CHECKPOINT_FILE" && return 0
    fi
    return 1
}

# Mark a step as completed in the checkpoint file.
mark_completed() {
    local step="$1"
    echo "$step:COMPLETED" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as completed"
}

# Mark a step as in-progress (removes any prior IN_PROGRESS marker first).
mark_in_progress() {
    local step="$1"
    if [ -f "$CHECKPOINT_FILE" ]; then
        # macOS BSD sed requires an explicit backup extension (empty string = no backup).
        # On GNU/Linux sed -i without extension also works, but '' is portable.
        sed -i '' "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE" 2>/dev/null \
            || sed -i "/^$step:IN_PROGRESS$/d" "$CHECKPOINT_FILE"
    fi
    echo "$step:IN_PROGRESS" >> "$CHECKPOINT_FILE"
    log "Marked step '$step' as in progress"
}

# Export environment variables required by fairseq / KenLM / Kaldi.
setup_path() {
    export HYDRA_FULL_ERROR=1
    export LD_LIBRARY_PATH="${KALDI_ROOT:-}/src/lib:${KENLM_ROOT}/lib:${LD_LIBRARY_PATH:-}"
}

# Activate the project's Python virtual environment.
activate_venv() {
    if [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
        log "Activating virtual environment at $VENV_PATH"
        source "$VENV_PATH/bin/activate"
    else
        log "[WARN] Virtual environment not found at $VENV_PATH — skipping activation."
    fi
}

# Create all pipeline output directories.
create_dirs() {
    mkdir -p "$MANIFEST_DIR" \
             "$CLUSTERING_DIR" \
             "$MANIFEST_NONSIL_DIR" \
             "$RESULTS_DIR" \
             "$CHECKPOINT_DIR" \
             "$LOG_DIR" \
             "$TEXT_OUTPUT" \
             "$GANS_OUTPUT_PHONES"
    log "Output directories created under $DATA_ROOT"
}
