#!/bin/bash

# =============================================================================
# wav2vec_functions.sh — Data-preparation pipeline functions
# =============================================================================
#
# CHANGES FROM ORIGINAL (see CHANGES.md for full rationale):
#   1. sed -i fixed for macOS.  BSD sed requires: sed -i '' 's/.../.../'.
#      GNU sed (Linux) accepts sed -i without a backup extension.
#      A portable wrapper (portable_sed_i) is used throughout.
#   2. TEST_RUN data limiting added to manifest creation functions.
#      When TEST_RUN=true (set in config.sh) only TEST_AUDIO_LIMIT audio
#      files per split are kept in each manifest, enabling a fast smoke-test
#      of the whole pipeline without processing the full 28 k-file dataset.
#   3. prepare_text uses TEST_TEXT_LIMIT to cap the text corpus size.
#   4. When TEST_RUN=false, AUDIO_DATA_PERCENT / TEXT_DATA_PERCENT optionally
#      subsample manifests and text (e.g. 20% full pipeline run).
#
# =============================================================================

set -e
set -o pipefail

# ── Accept dataset paths as positional arguments ──────────────────────────────
TRAIN_DATASETS=$1   # path to training WAV directory
VAL_DATASETS=$2     # path to validation WAV directory
TEST_DATASETS=$3    # path to test WAV directory
UNLABELLED_TEXT=$4  # path to unlabelled text file (one sentence per line)

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"


# ==================== HELPER FUNCTIONS ====================

# Portable in-place sed: macOS (BSD) sed requires sed -i '' '...'
# GNU sed (Linux) accepts sed -i '...'
portable_sed_i() {
    # $1 = expression   $2 = file
    if sed --version 2>&1 | grep -q GNU; then
        sed -i "$1" "$2"
    else
        sed -i '' "$1" "$2"
    fi
}


# Fix rVADfast's sflux() to return both spectral flatness and n_frames.
# The original sflux() only returned s_flatness; the modified vads.py
# expects both values.  This sed patch is applied once during pipeline init.
fixing_sflux() {
    local TARGET_FILE="$SPEECHPROCS"
    if [ -f "$TARGET_FILE" ]; then
        log "Patching sflux() return value in $TARGET_FILE ..."
        portable_sed_i '/def sflux/,/return/ s/^ *return .*/    return s_flatness, n_frames/' "$TARGET_FILE"
        log "sflux() patched successfully."
    else
        log "[ERROR] $TARGET_FILE not found — cannot patch sflux()"
        exit 1
    fi
}


# Patch add-self-loop-simple.cc: replace std::endl with "\n" to avoid
# a compatibility issue with pykaldi's stream handling.
replace_std_endl() {
    local input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        log "[ERROR] File not found: $input_file"
        return 1
    fi
    portable_sed_i 's/std::endl/"\\n"/g' "$input_file"
    log "Replaced std::endl with \"\\n\" in $input_file"
}


# Update --sample-pct in prepare_audio.sh (fraction of audio used for k-means).
update_sample_pct() {
    sed -i.bak -E \
        "s/(--sample-pct[[:space:]]+)[0-9]*\.?[0-9]+/\1${NEW_SAMPLE_PCT}/g" \
        "$PREPARE_AUDIO"
    log "Updated --sample-pct to ${NEW_SAMPLE_PCT} in prepare_audio.sh"
}


# Update --batch-size in prepare_audio.sh.
update_batch_size() {
    sed -i.bak -E \
        "s/(--batch-size[[:space:]]+)[0-9]+/\1${NEW_BATCH_SIZE}/g" \
        "$PREPARE_AUDIO"
    log "Updated --batch-size to ${NEW_BATCH_SIZE} in prepare_audio.sh"
}


# =============================================================================
# Helper: optionally truncate a manifest (TSV) for TEST_RUN only.
# Subsampling by AUDIO_DATA_PERCENT is done once in convert_audio.sh (FLAC list);
# manifests always list every WAV under data/wav — no second %-cut on TSVs.
# Manifests have a header line (the root path), so data rows are lines 2+.
# =============================================================================
maybe_truncate_manifest() {
    local manifest_file="$1"   # e.g. $MANIFEST_DIR/train.tsv
    local label="$2"           # human-readable name for logging

    local total_lines
    total_lines=$(wc -l < "$manifest_file")
    # Manifest line 1 is the header (root dir path), so data lines = total-1
    local data_lines=$(( total_lines - 1 ))
    if [ "$data_lines" -le 0 ]; then
        return 0
    fi

    if [ "$TEST_RUN" != true ]; then
        return 0
    fi

    local keep="$TEST_AUDIO_LIMIT"
    if [ "$data_lines" -le "$keep" ]; then
        log "[$label] TEST_RUN=true — manifest has $data_lines ≤ $keep entries — no truncation."
        return 0
    fi
    log "[$label] TEST_RUN=true — truncating manifest from $data_lines to $keep entries"

    local tmp_file="${manifest_file}.tmp"
    head -n $(( keep + 1 )) "$manifest_file" > "$tmp_file"
    mv "$tmp_file" "$manifest_file"
    log "[$label] Manifest truncated."
}


# ==================== MAIN PIPELINE STEPS ====================

# Step 1a: Create train manifest (train.tsv)
create_manifests_train() {
    local step_name="create_manifests_train"

    if is_completed "$step_name"; then
        log "Skipping train manifest creation (already completed)"
        return 0
    fi

    log "Creating TRAIN data manifest..."
    mark_in_progress "$step_name"

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TRAIN_DATASETS" \
        --dest "$MANIFEST_DIR" \
        --ext wav \
        --valid-percent 0

    if [ $? -eq 0 ]; then
        maybe_truncate_manifest "$MANIFEST_DIR/train.tsv" "train"
        mark_completed "$step_name"
        log "TRAIN manifest creation completed successfully"
    else
        log "[ERROR] TRAIN manifest creation failed"
        exit 1
    fi
}

# Step 1b: Create validation manifest (valid.tsv)
create_manifests_val() {
    local step_name="create_manifests_val"

    if is_completed "$step_name"; then
        log "Skipping validation manifest creation (already completed)"
        return 0
    fi

    log "Creating VALIDATION data manifest..."
    mark_in_progress "$step_name"

    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_DIR/val_manifest.XXXXXX")

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$VAL_DATASETS" \
        --dest "$TEMP_VAL_DIR" \
        --ext wav \
        --valid-percent 1.0

    local python_exit_code=$?

    if [ $python_exit_code -eq 0 ]; then
        if [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
            mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_DIR/valid.tsv"
            maybe_truncate_manifest "$MANIFEST_DIR/valid.tsv" "val"
            mark_completed "$step_name"
            log "VALIDATION manifest creation completed successfully"
        else
            log "[ERROR] Expected valid.tsv not found in $TEMP_VAL_DIR"
            rm -rf "$TEMP_VAL_DIR"
            exit 1
        fi
    else
        log "[ERROR] VALIDATION manifest creation failed (Python script error)"
        rm -rf "$TEMP_VAL_DIR"
        exit 1
    fi

    rm -rf "$TEMP_VAL_DIR"
}

# Step 1c: Create test manifest (placed into MANIFEST_NONSIL_DIR/test.tsv)
create_manifests_test() {
    local step_name="create_manifests_test"

    if is_completed "$step_name"; then
        log "Skipping test manifest creation (already completed)"
        return 0
    fi

    log "Creating TEST data manifest..."
    mark_in_progress "$step_name"

    local MANIFEST_TEST_DIR="$DATA_ROOT/manifest_test"
    mkdir -p "$MANIFEST_TEST_DIR"

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$TEST_DATASETS" \
        --dest "$MANIFEST_TEST_DIR" \
        --ext wav \
        --valid-percent 0

    maybe_truncate_manifest "$MANIFEST_TEST_DIR/train.tsv" "test"
    cp "$MANIFEST_TEST_DIR/train.tsv" "$MANIFEST_NONSIL_DIR/test.tsv"
    rm -rf "$MANIFEST_TEST_DIR"

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "TEST manifest creation completed successfully"
    else
        log "[ERROR] TEST manifest creation failed"
        exit 1
    fi
}


# Step 2: Run rVADfast to identify silence regions in each audio file.
create_rVADfast() {
    local step_name="create_rVADfast"

    fixing_sflux   # patch speechproc.py before running vads.py

    if is_completed "$step_name"; then
        log "Skipping rVADfast (already completed)"
        return 0
    fi

    log "Running rVADfast to detect silence ..."
    mark_in_progress "$step_name"

    python "$DIR_PATH/vads.py" -r "$RVAD_ROOT" < "$MANIFEST_DIR/train.tsv" > "$MANIFEST_DIR/train.vads"
    python "$DIR_PATH/vads.py" -r "$RVAD_ROOT" < "$MANIFEST_DIR/valid.tsv" > "$MANIFEST_DIR/valid.vads"

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "rVADfast completed successfully"
    else
        log "[ERROR] rVADfast failed"
        exit 1
    fi
}


# Step 3: Remove silence segments from audio using the .vads files.
remove_silence() {
    local step_name="remove_silence"

    if is_completed "$step_name"; then
        log "Skipping silence removal (already completed)"
        return 0
    fi

    log "Removing silence from audio ..."
    mark_in_progress "$step_name"

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" \
        --tsv "$MANIFEST_DIR/train.tsv" \
        --vads "$MANIFEST_DIR/train.vads" \
        --out "$NONSIL_AUDIO/train"

    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/remove_silence.py" \
        --tsv "$MANIFEST_DIR/valid.tsv" \
        --vads "$MANIFEST_DIR/valid.vads" \
        --out "$NONSIL_AUDIO/val"

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Silence removal completed successfully"
    else
        log "[ERROR] Silence removal failed"
        exit 1
    fi
}


# Step 4a: Re-create train manifest on silence-free audio.
create_manifests_nonsil_train() {
    local step_name="create_manifests_nonsil_train"

    if is_completed "$step_name"; then
        log "Skipping nonsil train manifest (already completed)"
        return 0
    fi

    log "Creating non-silence TRAIN manifest..."
    mark_in_progress "$step_name"

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/train" \
        --dest "$MANIFEST_NONSIL_DIR" \
        --ext wav \
        --valid-percent 0

    if [ $? -eq 0 ]; then
        maybe_truncate_manifest "$MANIFEST_NONSIL_DIR/train.tsv" "nonsil_train"
        mark_completed "$step_name"
        log "Nonsil TRAIN manifest completed successfully"
    else
        log "[ERROR] Nonsil TRAIN manifest failed"
        exit 1
    fi
}

# Step 4b: Re-create validation manifest on silence-free audio.
create_manifests_nonsil_val() {
    local step_name="create_manifests_nonsil_val"

    if is_completed "$step_name"; then
        log "Skipping nonsil val manifest (already completed)"
        return 0
    fi

    log "Creating non-silence VALIDATION manifest..."
    mark_in_progress "$step_name"

    local TEMP_VAL_DIR
    TEMP_VAL_DIR=$(mktemp -d "$MANIFEST_NONSIL_DIR/val_manifest.XXXXXX")

    python "$FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py" \
        "$NONSIL_AUDIO/val" \
        --dest "$TEMP_VAL_DIR" \
        --ext wav \
        --valid-percent 1.0
    local python_exit_code=$?

    if [ $python_exit_code -eq 0 ]; then
        if [ -f "$TEMP_VAL_DIR/valid.tsv" ]; then
            mkdir -p "$MANIFEST_NONSIL_DIR"
            mv "$TEMP_VAL_DIR/valid.tsv" "$MANIFEST_NONSIL_DIR/valid.tsv"
            maybe_truncate_manifest "$MANIFEST_NONSIL_DIR/valid.tsv" "nonsil_val"
            mark_completed "$step_name"
            log "Nonsil VALIDATION manifest completed successfully"
        else
            log "[ERROR] valid.tsv not found in $TEMP_VAL_DIR"
            rm -rf "$TEMP_VAL_DIR"
            exit 1
        fi
    else
        log "[ERROR] Nonsil VALIDATION manifest failed (Python exit code: $python_exit_code)"
        rm -rf "$TEMP_VAL_DIR"
        exit 1
    fi

    rm -rf "$TEMP_VAL_DIR"
}


# Step 5: Prepare audio features (k-means clustering → pseudo-phonemes).
prepare_audio() {
    local step_name="prepare_audio"
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"

    update_sample_pct
    update_batch_size

    if is_completed "$step_name"; then
        log "Skipping audio preparation (already completed)"
        return 0
    fi

    log "Preparing audio features (k-means clustering) ..."
    mark_in_progress "$step_name"

    # Faiss/OpenMP on macOS: k-means can deadlock with multi-threaded BLAS/OpenMP
    # (0% CPU).  Default to single-threaded here; raise FAISS_OMP_THREADS etc. if needed.
    export KMP_DUPLICATE_LIB_OK=TRUE
    if [ "$(uname -s)" = "Darwin" ]; then
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
        export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
        export FAISS_OMP_THREADS="${FAISS_OMP_THREADS:-1}"
    else
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    fi

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_audio.sh" \
        "$MANIFEST_NONSIL_DIR" "$CLUSTERING_DIR" "$MODEL" 512 14

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Audio preparation completed successfully"
    else
        log "[ERROR] Audio preparation failed"
        exit 1
    fi
}


# Step 6: Prepare text data (phonemise, build n-gram LM).
# When TEST_RUN=true, only TEST_TEXT_LIMIT lines. When TEST_RUN=false and
# TEXT_DATA_PERCENT < 100, use that percentage of lines (deterministic head).
prepare_text() {
    local step_name="prepare_text"
    export FAIRSEQ_ROOT="$FAIRSEQ_ROOT"
    export KENLM_ROOT="$KENLM_ROOT"

    if is_completed "$step_name"; then
        log "Skipping text preparation (already completed)"
        return 0
    fi

    log "Preparing text data ..."
    mark_in_progress "$step_name"

    # Determine which text file to pass to prepare_text.sh
    local text_file="$UNLABELLED_TEXT"
    if [ "$TEST_RUN" = true ]; then
        local subset_file="$DATA_ROOT/text_subset_${TEST_TEXT_LIMIT}.txt"
        if [ ! -f "$subset_file" ]; then
            log "TEST_RUN=true — creating text subset of $TEST_TEXT_LIMIT lines at $subset_file"
            head -n "$TEST_TEXT_LIMIT" "$UNLABELLED_TEXT" > "$subset_file"
        else
            log "TEST_RUN=true — reusing existing text subset $subset_file"
        fi
        text_file="$subset_file"
    elif [ "${TEXT_DATA_PERCENT:-100}" -lt 100 ] 2>/dev/null \
        && [ "${TEXT_DATA_PERCENT:-100}" -gt 0 ] 2>/dev/null; then
        local tpct="${TEXT_DATA_PERCENT:-100}"
        local subset_file="$DATA_ROOT/text_subset_pct${tpct}.txt"
        local total_lines
        total_lines=$(wc -l < "$UNLABELLED_TEXT")
        local keep=$(( total_lines * tpct / 100 ))
        if [ "$keep" -lt 1 ] && [ "$total_lines" -gt 0 ]; then keep=1; fi
        log "TEXT_DATA_PERCENT=$tpct — creating text subset of $keep / $total_lines lines at $subset_file"
        head -n "$keep" "$UNLABELLED_TEXT" > "$subset_file"
        text_file="$subset_file"
    fi

    replace_std_endl "$ADD_SELF_LOOP_SIMPLE"

    zsh "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/prepare_text.sh" \
        "$LANG" "$text_file" "$TEXT_OUTPUT" "$MIN_PHONES" "$PHONEMIZER" \
        "$FASTTEXT_LIB_MODEL" 0.25

    if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "Text preparation completed successfully"
    else
        log "[ERROR] Text preparation failed"
        exit 1
    fi
}
