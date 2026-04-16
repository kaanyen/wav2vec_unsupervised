#!/bin/bash

# This script runs the entire evaluation of the fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

# ==================== CONFIGURATION ====================
# Set these variables according to your environment and needs

source utils.sh

MODEL_PATH=$DIR_PATH/$1 # the model should be a .pt file

# Optional decode overrides (KenLM path): export before calling run_eval.sh
#   EVAL_BEAM EVAL_LM_WEIGHT EVAL_WORD_SCORE EVAL_BEAM_THRESHOLD

# ==================== HELPER FUNCTIONS ====================

_run_w2vu_generate() {
  local config_name=$1
  shift

  export HYDRA_FULL_ERROR=1
  export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
  export KENLM_ROOT="$KENLM_ROOT"
  export PYTHONPATH=$FAIRSEQ_ROOT:$PYTHONPATH

  CLUSTER_DATA="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
  GEN_SUBSET=test

  python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py" \
    --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate" \
    --config-name "$config_name" \
    fairseq.common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    fairseq.task.data="$CLUSTER_DATA" \
    fairseq.common_eval.path=$MODEL_PATH \
    results_path="$GANS_OUTPUT_PHONES" \
    fairseq.task.text_data="$TEXT_OUTPUT/phones/" \
    fairseq.dataset.batch_size=1 \
    fairseq.dataset.num_workers=0 \
    fairseq.dataset.required_batch_size_multiple=1 \
    fairseq.dataset.gen_subset="$GEN_SUBSET" \
    "$@"
}

#============================ model evaLuation =======================

transcription_gans_viterbi() {
  CLUSTER_DATA="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
  GEN_SUBSET=test

  OPTIONAL_TARGETS=()
  if [[ -f "$CLUSTER_DATA/${GEN_SUBSET}.phn" ]]; then
    OPTIONAL_TARGETS+=( "targets=phn" )
    echo "eval: found $CLUSTER_DATA/${GEN_SUBSET}.phn — passing targets=phn"
  fi
  PHONE_LM_BIN="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin"
  if [[ -f "$PHONE_LM_BIN" ]]; then
    OPTIONAL_TARGETS+=( "lm_model=$PHONE_LM_BIN" )
    echo "eval: found phone KenLM bin — passing lm_model=$PHONE_LM_BIN (for LM PPL / scoring)"
  fi

  _run_w2vu_generate viterbi "${OPTIONAL_TARGETS[@]}"
}

# KenLM + Flashlight beam search (requires flashlight-text + flashlight-sequence).
# Default: phone-unit LM (matches Wav2Vec-U phoneme targets). Override with EVAL_KENLM_MODE=words.
transcription_gans_kenlm() {
  CLUSTER_DATA="$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled"
  GEN_SUBSET=test

  MODE="${EVAL_KENLM_MODE:-phones}"
  OPTIONAL_TARGETS=(
    "beam=${EVAL_BEAM:-10}"
    "lm_weight=${EVAL_LM_WEIGHT:-0}"
    "word_score=${EVAL_WORD_SCORE:-1.0}"
    "beam_threshold=${EVAL_BEAM_THRESHOLD:-50.0}"
    "beam_size_token=${EVAL_BEAM_SIZE_TOKEN:-500}"
  )

  if [[ "$MODE" == "words" ]]; then
    LEXICON="$TEXT_OUTPUT/lexicon_filtered.lst"
    WORD_LM="$TEXT_OUTPUT/kenlm.wrd.o40003.bin"
    if [[ ! -f "$LEXICON" || ! -f "$WORD_LM" ]]; then
      echo "eval: ERROR words mode needs $LEXICON and $WORD_LM"
      exit 1
    fi
    OPTIONAL_TARGETS+=( "unit_lm=false" "lexicon=$LEXICON" "lm_model=$WORD_LM" )
    echo "eval: KenLM (word LM + lexicon) beam=${EVAL_BEAM:-10} lm_weight=${EVAL_LM_WEIGHT:-0}"
  else
    PHONE_LM_BIN="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin"
    if [[ ! -f "$PHONE_LM_BIN" ]]; then
      echo "eval: ERROR missing phone KenLM: $PHONE_LM_BIN"
      exit 1
    fi
    OPTIONAL_TARGETS+=( "unit_lm=true" "lm_model=$PHONE_LM_BIN" )
    echo "eval: KenLM (phone unit LM) lm=$PHONE_LM_BIN beam=${EVAL_BEAM:-10} lm_weight=${EVAL_LM_WEIGHT:-0}"
  fi

  if [[ -f "$CLUSTER_DATA/${GEN_SUBSET}.phn" ]]; then
    OPTIONAL_TARGETS+=( "targets=phn" )
    echo "eval: targets=phn from $CLUSTER_DATA/${GEN_SUBSET}.phn"
  fi

  _run_w2vu_generate kenlm "${OPTIONAL_TARGETS[@]}"
}
