#!/usr/bin/env bash
# Sweep KenLM decode settings (no GAN retraining). Default: phone-unit LM (see EVAL_KENLM_MODE).
# Usage:
#   ./scripts/sweep_kenlm_decode.sh outputs/2026-04-10/14-31-59/checkpoint_best.pt
# Word+lexicon mode:
#   EVAL_KENLM_MODE=words ./scripts/sweep_kenlm_decode.sh outputs/.../checkpoint_best.pt

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="${1:?Usage: $0 <checkpoint.pt relative to repo>}"
OUT_DIR="$ROOT/data/results/librispeech"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/kenlm_sweep_$(date +%Y%m%d_%H%M%S).tsv"
MODE="${EVAL_KENLM_MODE:-phones}"
echo -e "mode\tbeam\tlm_weight\tword_score\twer\tlm_ppl\ttokens_line" | tee "$SUMMARY"

for beam in 5 10 20; do
  for lm_w in 0 0.1 0.2 0.25; do
    for ws in 0.5 1.0; do
      echo "=== mode=$MODE beam=$beam lm_weight=$lm_w word_score=$ws ===" >&2
      LOG=$(mktemp)
      if EVAL_KENLM_MODE="$MODE" EVAL_BEAM="$beam" EVAL_LM_WEIGHT="$lm_w" EVAL_WORD_SCORE="$ws" \
        ./run_eval.sh "$CKPT" kenlm >"$LOG" 2>&1; then
        wer=$(grep '\[INFO\] - WER:' "$LOG" | tail -1 | sed 's/.*WER: //' || echo "NA")
        ppl=$(grep '\[INFO\] - LM PPL:' "$LOG" | tail -1 | sed 's/.*LM PPL: //' || echo "NA")
        tok=$(grep 'length:' "$LOG" | tail -1 | sed 's/.*length: //' | awk '{print $1}' || echo "NA")
        echo -e "${MODE}\t${beam}\t${lm_w}\t${ws}\t${wer}\t${ppl}\t${tok}" | tee -a "$SUMMARY"
      else
        echo -e "${MODE}\t${beam}\t${lm_w}\t${ws}\tFAILED\tFAILED\tFAILED" | tee -a "$SUMMARY"
        tail -30 "$LOG" >&2 || true
      fi
      rm -f "$LOG"
    done
  done
done

echo "Summary: $SUMMARY" >&2
