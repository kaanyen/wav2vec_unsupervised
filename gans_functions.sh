#!/bin/bash

# This script runs the GANS training of  unsupervised wav2vec pipeline

# Wav2Vec Unsupervised Pipeline Runner
# This script runs the entire fairseq wav2vec unsupervised pipeline
# with checkpointing to allow resuming from any step

set -e  # Exit on error
set -o pipefail  # Exit if any command in a pipe fails

source utils.sh

# =============================================================================
# prepare_gan_train_subset  — optionally create a lightweight train split
# =============================================================================
# When GAN_TRAIN_UTTERANCES > 0, build a subset directory beside the original
# clustering dir that contains:
#   train.tsv     — first N+1 lines  (header + N utterance rows)
#   train.lengths — first N lines    (one length per utterance)
#   train.npy     — symlink to the original (dataset uses mmap offsets, so the
#                   full file can be shared with no copying)
#   valid.*  test.* — copied unchanged
#
# Returns (via stdout) the path that should be passed to fairseq as task.data.
# =============================================================================
prepare_gan_train_subset() {
    local src_dir="$1"  # e.g. $CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled
    local n="${GAN_TRAIN_UTTERANCES:-0}"

    if [ "$n" -le 0 ] 2>/dev/null; then
        echo "$src_dir"
        return 0
    fi

    local dst_dir="${src_dir}_sub${n}"

    if [ -d "$dst_dir" ] && [ -f "$dst_dir/train.tsv" ]; then
        local existing
        existing=$(( $(wc -l < "$dst_dir/train.tsv") - 1 ))
        if [ "$existing" -eq "$n" ]; then
            log "[subset] Reusing existing ${n}-utterance subset at $dst_dir" >&2
            echo "$dst_dir"
            return 0
        fi
    fi

    log "[subset] Creating ${n}-utterance training subset → $dst_dir" >&2
    mkdir -p "$dst_dir"

    # Truncate train manifest and lengths; symlink the large npy (mmap-safe)
    head -n $(( n + 1 )) "$src_dir/train.tsv"     > "$dst_dir/train.tsv"
    head -n "$n"          "$src_dir/train.lengths" > "$dst_dir/train.lengths"
    ln -sf "$src_dir/train.npy" "$dst_dir/train.npy"

    # Copy valid and test splits unchanged
    for split in valid test; do
        for ext in tsv npy lengths; do
            [ -f "$src_dir/${split}.${ext}" ] && \
                cp -f "$src_dir/${split}.${ext}" "$dst_dir/${split}.${ext}"
        done
    done

    local actual
    actual=$(( $(wc -l < "$dst_dir/train.tsv") - 1 ))
    log "[subset] Done — ${actual} utterances in $dst_dir" >&2
    echo "$dst_dir"
}

#=========================== GANS training and preparation ==============================
train_gans(){
   local step_name="train_gans"
   export FAIRSEQ_ROOT=$FAIRSEQ_ROOT
   # export KALDI_ROOT="$DIR_PATH/pykaldi/tools/kaldi"
   export KENLM_ROOT="$KENLM_ROOT"
   export PYTHONPATH="${DIR_PATH}:${PYTHONPATH:-}"


   if is_completed "$step_name"; then
        log "Skipping gans training  (already completed)"
        return 0
    fi

    log "gans training."
    mark_in_progress "$step_name"

    # Wire config.sh DEVICE into Hydra (Wav2VecU reads model.device in wav2vec_u.py).
    local gan_cpu_flag="common.cpu=false"
    local gan_model_device="+model.device=auto"
    case "${DEVICE:-auto}" in
      cpu)
        gan_cpu_flag="common.cpu=true"
        gan_model_device="+model.device=cpu"
        ;;
      mps|cuda)
        gan_cpu_flag="common.cpu=false"
        gan_model_device="+model.device=${DEVICE}"
        ;;
      auto|*)
        gan_cpu_flag="common.cpu=false"
        gan_model_device="+model.device=auto"
        ;;
    esac
    log "GAN device: DEVICE=$DEVICE → ${gan_model_device#+}, ${gan_cpu_flag}"
    log "GAN fairseq: num_workers=${GAN_NUM_WORKERS:-1} max_update=${GAN_MAX_UPDATE:-5000} validate/save every ${GAN_VALIDATE_INTERVAL_UPDATES:-1000} updates"

    # Optionally limit training utterances (0 = use all).
    local train_data_dir
    train_data_dir=$(prepare_gan_train_subset \
        "$CLUSTERING_DIR/precompute_pca512_cls128_mean_pooled")
    log "GAN task.data → $train_data_dir"

   # Single run (no Hydra -m multirun). Dataset/optimization intervals from config.sh (GAN_*).
   PYTHONPATH=$FAIRSEQ_ROOT PREFIX=w2v_unsup_gan_xp fairseq-hydra-train \
    --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name w2vu \
    task.data="$train_data_dir" \
    task.text_data="$TEXT_OUTPUT/phones/" \
    task.kenlm_path="$TEXT_OUTPUT/phones/lm.phones.filtered.04.bin" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    ${gan_cpu_flag} \
    ${gan_model_device} \
    common.log_format=simple \
    common.log_interval="${GAN_LOG_INTERVAL:-5}" \
    dataset.num_workers="${GAN_NUM_WORKERS:-1}" \
    optimization.max_update="${GAN_MAX_UPDATE:-5000}" \
    dataset.validate_interval="${GAN_VALIDATE_INTERVAL_UPDATES:-1000}" \
    dataset.validate_interval_updates="${GAN_VALIDATE_INTERVAL_UPDATES:-1000}" \
    checkpoint.save_interval="${GAN_SAVE_INTERVAL_UPDATES:-1000}" \
    checkpoint.save_interval_updates="${GAN_SAVE_INTERVAL_UPDATES:-1000}" \
    model.code_penalty=6 model.gradient_penalty=0.5 \
    model.smoothness_weight=1.5 common.seed=0 \
    +optimizer.groups.generator.optimizer.lr="[0.00004]" \
    +optimizer.groups.discriminator.optimizer.lr="[0.00002]" \
    ~optimizer.groups.generator.optimizer.amsgrad \
    ~optimizer.groups.discriminator.optimizer.amsgrad \
    2>&1 | tee $RESULTS_DIR/training1.log

    

   if [ $? -eq 0 ]; then
        mark_completed "$step_name"
        log "gans trained successfully"
    else
        log "ERROR: gans training failed"
        exit 1
    fi
}

