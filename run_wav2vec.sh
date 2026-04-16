#!/bin/bash

# =============================================================================
# run_wav2vec.sh — Data preparation pipeline (manifests, silence, features)
# =============================================================================
#
# CHANGES FROM ORIGINAL (see CHANGES.md for full rationale):
#   1. config.sh sourced so DEVICE, TEST_RUN, and TEST_AUDIO_LIMIT are
#      available to all called functions.
#   2. convert_audio.sh called first to convert LibriSpeech FLAC → WAV 16kHz.
#      The original scripts assumed WAV input; LibriSpeech ships as FLAC.
#   3. WAV paths (data/wav/{train,val,test}) passed to wav2vec_functions as
#      positional arguments instead of the raw FLAC paths.
#   4. Unlabelled text path defaults to the wikitext corpus in data/corpora/.
#   5. All steps logged to data/logs/librispeech/pipeline.log via utils.sh.
#
# USAGE (two calling modes)
#
#   Mode A — use default data paths (LibriSpeech + wikitext in data/corpora/):
#       ./run_wav2vec.sh
#
#   Mode B — supply custom paths explicitly (same as original interface):
#       ./run_wav2vec.sh "/path/to/train_wav" \
#                        "/path/to/val_wav"   \
#                        "/path/to/test_wav"  \
#                        "/path/to/text.txt"
#
# =============================================================================

set -e
set -o pipefail

# ── Locate project root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load central config (DEVICE, TEST_RUN, …) ────────────────────────────────
source "$SCRIPT_DIR/config.sh"

# ── Default data paths ────────────────────────────────────────────────────────
# These match the actual data layout in data/corpora/.
# Paths will be overridden if the user passes positional arguments.
DEFAULT_TRAIN_WAV="$SCRIPT_DIR/data/wav/train"
DEFAULT_VAL_WAV="$SCRIPT_DIR/data/wav/val"
DEFAULT_TEST_WAV="$SCRIPT_DIR/data/wav/test"
DEFAULT_TEXT="$SCRIPT_DIR/data/corpora/wikitext_103_raw_v1.txt"

# ── Accept optional custom paths (original interface preserved) ───────────────
TRAIN_IN="${1:-$DEFAULT_TRAIN_WAV}"
VAL_IN="${2:-$DEFAULT_VAL_WAV}"
TEST_IN="${3:-$DEFAULT_TEST_WAV}"
TEXT_IN="${4:-$DEFAULT_TEXT}"

# ── Load pipeline functions (must come after positional args are set) ─────────
source "$SCRIPT_DIR/wav2vec_functions.sh" "$TRAIN_IN" "$VAL_IN" "$TEST_IN" "$TEXT_IN"

# ── Create output directories and activate venv ───────────────────────────────
create_dirs
activate_venv
setup_path

log "====================================================="
log " wav2vec-U  Data Preparation  —  $(date)"
log " DEVICE    : $DEVICE"
log " TEST_RUN  : $TEST_RUN  (limit=$TEST_AUDIO_LIMIT audio / $TEST_TEXT_LIMIT text)"
log " Data %    : AUDIO_DATA_PERCENT=${AUDIO_DATA_PERCENT:-100}  TEXT_DATA_PERCENT=${TEXT_DATA_PERCENT:-100}"
log " Train data: $TRAIN_IN"
log " Val data  : $VAL_IN"
log " Test data : $TEST_IN"
log " Text data : $TEXT_IN"
log "====================================================="

# ── LibriSpeech FLAC roots (must match convert_audio.sh) ─────────────────────
LIBRISPEECH_FLAC_ROOT="$SCRIPT_DIR/data/corpora/LibriSpeech"
TRAIN_FLAC_DIR="$LIBRISPEECH_FLAC_ROOT/train-clean-100"

# Count *.flac under a directory (0 if missing)
_count_flac_files() {
    local d="$1"
    if [ ! -d "$d" ]; then
        echo 0
        return
    fi
    find "$d" -name "*.flac" 2>/dev/null | wc -l | tr -d ' '
}

# Minimum train WAVs needed for current config (same rules as convert_audio.sh).
_required_train_wavs_from_flac() {
    local nflac
    nflac=$(_count_flac_files "$TRAIN_FLAC_DIR")
    if [ "${nflac:-0}" -eq 0 ]; then
        echo 0
        return
    fi
    if [ "$TEST_RUN" = true ]; then
        local lim=$TEST_AUDIO_LIMIT
        if [ "$nflac" -lt "$lim" ]; then lim=$nflac; fi
        echo "$lim"
    elif [ "${AUDIO_DATA_PERCENT:-100}" -lt 100 ] 2>/dev/null \
        && [ "${AUDIO_DATA_PERCENT:-100}" -gt 0 ] 2>/dev/null; then
        local need=$(( nflac * AUDIO_DATA_PERCENT / 100 ))
        if [ "$need" -lt 1 ]; then need=1; fi
        echo "$need"
    else
        echo "$nflac"
    fi
}

# =============================================================================
# STEP 0: Convert FLAC → WAV (only needed if using the default LibriSpeech data)
# =============================================================================
# If custom WAV paths were supplied (Mode B), skip conversion.
# Otherwise: skip only when data/wav/train has at least as many WAVs as this
# config requires (from LibriSpeech FLAC count × TEST_RUN / AUDIO_DATA_PERCENT).
# A leftover smoke-test folder (e.g. 30 WAVs) is insufficient for 20% / full runs
# and will trigger conversion to read the real corpus under data/corpora/LibriSpeech.
if [ "${1+set}" != "set" ]; then
    TRAIN_WAV_COUNT=0
    if [ -d "$DEFAULT_TRAIN_WAV" ]; then
        TRAIN_WAV_COUNT=$(find "$DEFAULT_TRAIN_WAV" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
    fi
    REQUIRED_TRAIN=$(_required_train_wavs_from_flac)
    if [ ! -d "$TRAIN_FLAC_DIR" ]; then
        log "[WARN] LibriSpeech train FLAC dir not found: $TRAIN_FLAC_DIR"
    fi
    if [ "$TRAIN_WAV_COUNT" -eq 0 ]; then
        log "No WAV files in $DEFAULT_TRAIN_WAV — running FLAC→WAV conversion from $LIBRISPEECH_FLAC_ROOT ..."
        bash "$SCRIPT_DIR/convert_audio.sh"
    elif [ "${REQUIRED_TRAIN:-0}" -gt 0 ] && [ "$TRAIN_WAV_COUNT" -lt "$REQUIRED_TRAIN" ]; then
        log "Train WAVs ($TRAIN_WAV_COUNT) < required for this config ($REQUIRED_TRAIN from corpus) — running FLAC→WAV conversion ..."
        bash "$SCRIPT_DIR/convert_audio.sh"
    else
        log "WAV cache OK ($TRAIN_WAV_COUNT train WAVs, need ≥ $REQUIRED_TRAIN) — skipping conversion."
    fi
fi

# =============================================================================
# STEP 1: Create data manifests (TSV files listing audio paths + lengths)
# =============================================================================
log "Creating audio manifests..."
create_manifests_train 0
create_manifests_val 0
create_manifests_test 0

# =============================================================================
# STEP 2: Voice Activity Detection — find silence regions
# =============================================================================
create_rVADfast

# =============================================================================
# STEP 3: Remove silence from audio files
# =============================================================================
remove_silence

# =============================================================================
# STEP 4: Re-create manifests on silence-free audio
# =============================================================================
create_manifests_nonsil_train 0.1
create_manifests_nonsil_val 0.1

# =============================================================================
# STEP 5: Prepare audio features (wav2vec 2.0 embeddings + k-means clustering)
# =============================================================================
prepare_audio

# =============================================================================
# STEP 6: Prepare text (phonemise + build KenLM language model)
# =============================================================================
prepare_text

log "====================================================="
log " Data preparation completed successfully!"
log " Next: run ./run_gans.sh to train the GAN."
log "====================================================="
