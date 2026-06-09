#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
if [ "$(basename "${ROOT_DIR}")" = "scripts" ]; then
  ROOT_DIR="$(dirname "${ROOT_DIR}")"
fi
cd "${ROOT_DIR}"

PHASE=${PHASE:-all}          # train_extra / eval / all
POOLS=${POOLS:-"sum"}        # use "sum mean" if you also want mean
SEEDS=${SEEDS:-"0 1 2"}
SETTINGS=${SETTINGS:-"10+1000@100 10+10000@100"}
MAX_EPOCHS=${MAX_EPOCHS:-300}
LOG_ROOT=${LOG_ROOT:-logs-pooling-sensitivity}

mkdir -p "${LOG_ROOT}"
mkdir -p checkpoints/PoolingSensitivity

latest_ckpt() {
  find outputs -path '*/checkpoints/*.ckpt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

train_extra_pools() {
  for pool in ${POOLS}; do
    for seed in ${SEEDS}; do
      echo "============================================================"
      echo "Training Embedding-only pool=${pool}, seed=${seed}"
      echo "============================================================"

      python -m main +name=PoolSens_embedding_${pool}_pre_seed${seed} \
        dataset=elliptic_recommendation \
        algorithm=iterative_filtering \
        algorithm.use_anchor_features=false \
        algorithm.model.pool=${pool} \
        experiment=exp_edge_recommendation \
        'experiment.tasks=[training]' \
        experiment.validation.test_during_training=false \
        experiment.training.max_epochs=${MAX_EPOCHS} \
        seed=${seed} \
        wandb.mode=offline

      pre_ckpt=$(latest_ckpt)
      cp "${pre_ckpt}" checkpoints/PoolingSensitivity/embedding_${pool}_pre_seed${seed}.ckpt

      python -m main +name=PoolSens_embedding_${pool}_tune_seed${seed} \
        dataset=elliptic_recommendation \
        dataset.augment.enabled=true \
        algorithm=iterative_filtering \
        algorithm.use_anchor_features=false \
        algorithm.model.pool=${pool} \
        experiment=exp_edge_recommendation \
        'experiment.tasks=[training]' \
        experiment.validation.test_during_training=false \
        experiment.training.early_stopping.enabled=false \
        experiment.training.max_epochs=${MAX_EPOCHS} \
        seed=${seed} \
        wandb.mode=offline \
        load=checkpoints/PoolingSensitivity/embedding_${pool}_pre_seed${seed}.ckpt

      tuned_ckpt=$(latest_ckpt)
      cp "${tuned_ckpt}" checkpoints/PoolingSensitivity/embedding_${pool}_tuned_seed${seed}.ckpt

      echo "============================================================"
      echo "Training Full-SAIF pool=${pool}, seed=${seed}"
      echo "============================================================"

      python -m main +name=PoolSens_saif_${pool}_pre_seed${seed} \
        dataset=elliptic_recommendation \
        algorithm=iterative_filtering \
        algorithm.use_anchor_features=true \
        algorithm.model.anchor_feature_mode=full \
        algorithm.model.anchor_fusion_mode=full \
        algorithm.model.pool=${pool} \
        experiment=exp_edge_recommendation \
        'experiment.tasks=[training]' \
        experiment.validation.test_during_training=false \
        experiment.training.max_epochs=${MAX_EPOCHS} \
        seed=${seed} \
        wandb.mode=offline

      pre_ckpt=$(latest_ckpt)
      cp "${pre_ckpt}" checkpoints/PoolingSensitivity/saif_${pool}_pre_seed${seed}.ckpt

      python -m main +name=PoolSens_saif_${pool}_tune_seed${seed} \
        dataset=elliptic_recommendation \
        dataset.augment.enabled=true \
        algorithm=iterative_filtering \
        algorithm.use_anchor_features=true \
        algorithm.model.anchor_feature_mode=full \
        algorithm.model.anchor_fusion_mode=full \
        algorithm.model.pool=${pool} \
        experiment=exp_edge_recommendation \
        'experiment.tasks=[training]' \
        experiment.validation.test_during_training=false \
        experiment.training.early_stopping.enabled=false \
        experiment.training.max_epochs=${MAX_EPOCHS} \
        seed=${seed} \
        wandb.mode=offline \
        load=checkpoints/PoolingSensitivity/saif_${pool}_pre_seed${seed}.ckpt

      tuned_ckpt=$(latest_ckpt)
      cp "${tuned_ckpt}" checkpoints/PoolingSensitivity/saif_${pool}_tuned_seed${seed}.ckpt
    done
  done
}

eval_all() {
  for seed in ${SEEDS}; do
    for pool in max ${POOLS}; do
      for variant in embedding saif; do
        if [[ "${pool}" == "max" && "${variant}" == "embedding" ]]; then
          ckpt="checkpoints/RevTrack/${seed}_tuned.ckpt"
        elif [[ "${pool}" == "max" && "${variant}" == "saif" ]]; then
          ckpt="checkpoints/AnchorRevFilter/tuned_seed${seed}.ckpt"
        elif [[ "${variant}" == "embedding" ]]; then
          ckpt="checkpoints/PoolingSensitivity/embedding_${pool}_tuned_seed${seed}.ckpt"
        else
          ckpt="checkpoints/PoolingSensitivity/saif_${pool}_tuned_seed${seed}.ckpt"
        fi

        for setting in ${SETTINGS}; do
          tag=${setting//+/p}
          tag=${tag//@/at}

          if [[ "${variant}" == "saif" ]]; then
            python -m main +name=PoolSens_${variant}_${pool}_ckpt${seed}_${tag} \
              dataset=elliptic_recommendation \
              dataset.eval_pool_mode=official \
              algorithm=iterative_filtering \
              algorithm.use_anchor_features=true \
              algorithm.model.anchor_feature_mode=full \
              algorithm.model.anchor_fusion_mode=full \
              algorithm.model.pool=${pool} \
              experiment=exp_edge_recommendation \
              'experiment.tasks=[test]' \
              experiment.test.batch_size=16 \
              seed=0 \
              wandb.mode=offline \
              load=${ckpt} \
              +shortcut="${setting}" \
              2>&1 | tee "${LOG_ROOT}/${variant}_${pool}_ckpt${seed}_${tag}.log"
          else
            python -m main +name=PoolSens_${variant}_${pool}_ckpt${seed}_${tag} \
              dataset=elliptic_recommendation \
              dataset.eval_pool_mode=official \
              algorithm=iterative_filtering \
              algorithm.use_anchor_features=false \
              algorithm.model.pool=${pool} \
              experiment=exp_edge_recommendation \
              'experiment.tasks=[test]' \
              experiment.test.batch_size=16 \
              seed=0 \
              wandb.mode=offline \
              load=${ckpt} \
              +shortcut="${setting}" \
              2>&1 | tee "${LOG_ROOT}/${variant}_${pool}_ckpt${seed}_${tag}.log"
          fi
        done
      done
    done
  done
}

if [[ "${PHASE}" == "train_extra" || "${PHASE}" == "all" ]]; then
  train_extra_pools
fi

if [[ "${PHASE}" == "eval" || "${PHASE}" == "all" ]]; then
  eval_all
fi
