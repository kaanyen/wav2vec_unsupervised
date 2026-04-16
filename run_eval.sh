#!/bin/bash

# Source the function definitions

source "$(dirname "$0")/eval_functions.sh"

create_dirs
activate_venv

# Usage:
#   ./run_eval.sh <path_to_checkpoint.pt>              # Viterbi + optional phone LM (default)
#   ./run_eval.sh <path_to_checkpoint.pt> kenlm       # Lexicon + word KenLM beam decode
#
# KenLM tuning (optional env vars, only for kenlm mode):
#   EVAL_KENLM_MODE=phones|words   (default phones = unit phone LM + lm.phones.filtered.04.bin)
#   EVAL_BEAM EVAL_LM_WEIGHT (default 0; raise slowly — high values often empty the beam)
#   EVAL_WORD_SCORE EVAL_BEAM_THRESHOLD EVAL_BEAM_SIZE_TOKEN

CHECKPOINT_REL="${1:?Usage: $0 <checkpoint.pt> [viterbi|kenlm]}"
DECODER_MODE="${2:-viterbi}"

case "$DECODER_MODE" in
  kenlm|KENLM)
    transcription_gans_kenlm
    ;;
  viterbi|VITERBI|"")
    transcription_gans_viterbi
    ;;
  *)
    echo "Unknown decoder mode: $DECODER_MODE (use viterbi or kenlm)"
    exit 1
    ;;
esac
