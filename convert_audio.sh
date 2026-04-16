#!/bin/bash

# =============================================================================
# convert_audio.sh — Convert LibriSpeech FLAC files to 16 kHz WAV
# =============================================================================
#
# WHY THIS EXISTS
#   LibriSpeech ships audio in FLAC format.  The fairseq wav2vec manifest
#   scripts (wav2vec_manifest.py) and the remove_silence pipeline expect
#   plain WAV files at 16 kHz mono.  This script uses ffmpeg (available
#   via Homebrew on macOS) to do the conversion.
#
# OUTPUTS
#   $INSTALL_ROOT/data/wav/train/   — training audio (WAV, 16 kHz mono)
#   $INSTALL_ROOT/data/wav/val/     — validation audio
#   $INSTALL_ROOT/data/wav/test/    — test audio
#
# DATA LIMITS (config.sh)
#   TEST_RUN=true → first TEST_AUDIO_LIMIT FLACs per split (smoke test).
#   TEST_RUN=false and AUDIO_DATA_PERCENT < 100 → first fraction of FLACs per split.
#
# USAGE
#   Called automatically by run_wav2vec.sh, or run standalone:
#       source config.sh && bash convert_audio.sh
#
# =============================================================================

set -e
set -o pipefail

# ── Locate the project root (works however the script is invoked) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source config and utils for logging + variables ───────────────────────────
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/utils.sh"

# ── Verify ffmpeg is installed ────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    log "[ERROR] ffmpeg not found. Install it with:  brew install ffmpeg"
    exit 1
fi

# ── Source directories (LibriSpeech FLAC) ────────────────────────────────────
LIBRISPEECH_ROOT="$SCRIPT_DIR/data/corpora/LibriSpeech"
TRAIN_FLAC_DIR="$LIBRISPEECH_ROOT/train-clean-100"
VAL_FLAC_DIR="$LIBRISPEECH_ROOT/dev-clean"
TEST_FLAC_DIR="$LIBRISPEECH_ROOT/test-clean"

# ── Destination directories (WAV) ────────────────────────────────────────────
WAV_ROOT="$SCRIPT_DIR/data/wav"
TRAIN_WAV_DIR="$WAV_ROOT/train"
VAL_WAV_DIR="$WAV_ROOT/val"
TEST_WAV_DIR="$WAV_ROOT/test"

mkdir -p "$TRAIN_WAV_DIR" "$VAL_WAV_DIR" "$TEST_WAV_DIR"

# =============================================================================
# Helper: convert one split
#   $1 = source directory containing nested FLAC files
#   $2 = destination directory for flat WAV files
#   $3 = human-readable split name (for logging)
# =============================================================================
convert_split() {
    local src_dir="$1"
    local dst_dir="$2"
    local split_name="$3"

    log "[$split_name] Scanning FLAC files in $src_dir ..."

    # Collect all .flac paths into an array (no `mapfile` — macOS ships Bash 3.2,
    # which does not support mapfile/readarray; those require Bash 4+).
    local flac_files=()
    local line
    while IFS= read -r line; do
        flac_files+=("$line")
    done < <(find "$src_dir" -name "*.flac" | sort)

    local total=${#flac_files[@]}

    if [ "$total" -eq 0 ]; then
        log "[ERROR] No .flac files found in $src_dir"
        exit 1
    fi

    # Apply test-run or fractional limit (manual slice — Bash 3.2-safe)
    local limit_n="$total"
    if [ "$TEST_RUN" = true ]; then
        if [ "$total" -gt "$TEST_AUDIO_LIMIT" ]; then
            limit_n="$TEST_AUDIO_LIMIT"
            log "[$split_name] TEST_RUN=true — limiting to $limit_n / $total files"
        else
            log "[$split_name] TEST_RUN=true — converting all $total files (≤ TEST_AUDIO_LIMIT)"
        fi
    elif [ "${AUDIO_DATA_PERCENT:-100}" -lt 100 ] 2>/dev/null \
        && [ "${AUDIO_DATA_PERCENT:-100}" -gt 0 ] 2>/dev/null; then
        local apct="${AUDIO_DATA_PERCENT:-100}"
        limit_n=$(( total * apct / 100 ))
        if [ "$limit_n" -lt 1 ]; then limit_n=1; fi
        if [ "$limit_n" -lt "$total" ]; then
            log "[$split_name] AUDIO_DATA_PERCENT=$apct — using $limit_n / $total FLAC files"
        else
            log "[$split_name] AUDIO_DATA_PERCENT=$apct — all $total files (fraction ≤ corpus)"
        fi
    else
        log "[$split_name] Converting all $total files"
    fi

    if [ "$limit_n" -lt "$total" ]; then
        local limited=()
        local i=0
        for line in "${flac_files[@]}"; do
            [ "$i" -ge "$limit_n" ] && break
            limited+=("$line")
            i=$((i + 1))
        done
        flac_files=("${limited[@]}")
        total=$limit_n
    fi

    local converted=0
    local skipped=0

    for flac_path in "${flac_files[@]}"; do
        # Derive a flat filename: replace path separators with underscores
        # e.g. 1272/128104/1272-128104-0000.flac → 1272-128104-0000.wav
        local basename
        basename="$(basename "$flac_path" .flac).wav"
        local wav_path="$dst_dir/$basename"

        if [ -f "$wav_path" ]; then
            (( skipped++ )) || true
            continue
        fi

        # Convert: 16 kHz, mono, 16-bit PCM WAV
        ffmpeg -hide_banner -loglevel error \
               -i "$flac_path" \
               -ar 16000 -ac 1 -sample_fmt s16 \
               "$wav_path"

        (( converted++ )) || true

        # Progress every 100 files
        if (( converted % 100 == 0 )); then
            log "[$split_name] Converted $converted / $total ..."
        fi
    done

    log "[$split_name] Done. Converted=$converted  Skipped(already exist)=$skipped"
}

# =============================================================================
# Main
# =============================================================================
log "===== Audio Conversion: FLAC → WAV (16 kHz mono) ====="
log "TEST_RUN=$TEST_RUN   TEST_AUDIO_LIMIT=$TEST_AUDIO_LIMIT   AUDIO_DATA_PERCENT=${AUDIO_DATA_PERCENT:-100}"

convert_split "$TRAIN_FLAC_DIR" "$TRAIN_WAV_DIR" "train"
convert_split "$VAL_FLAC_DIR"   "$VAL_WAV_DIR"   "val"
convert_split "$TEST_FLAC_DIR"  "$TEST_WAV_DIR"  "test"

log "===== Conversion complete ====="
log "WAV files written to: $WAV_ROOT"

# Export paths so run_wav2vec.sh can pick them up directly
export CONVERTED_TRAIN_WAV="$TRAIN_WAV_DIR"
export CONVERTED_VAL_WAV="$VAL_WAV_DIR"
export CONVERTED_TEST_WAV="$TEST_WAV_DIR"
