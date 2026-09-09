#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-lineage_fixed_20260530_2gpu}"
BASE_OUT="/data/yanjie_huang/eva/EVA1/data/position_ablation_eukaryote/output_lineage_fixed/${RUN_NAME}"
MOUNT_OUT="/eva/data/position_ablation_eukaryote/output_lineage_fixed/${RUN_NAME}"
DATASET="/eva/data/position_ablation_eukaryote/delta_ll_results_all_data.json"
SCRIPT="/eva/scripts/position_ablation_eukaryote/eva_position_ablation_eukaryote_lineage_fixed.py"
MERGE_SCRIPT="/eva/scripts/position_ablation_eukaryote/merge_position_ablation_shards.py"
IMAGE="eva:latest"
HOST_EVA="/data/yanjie_huang/eva/EVA1"

mkdir -p "${BASE_OUT}"/{logs,eva1400M_shard0of2,eva1400M_shard1of2,eva1400M_merged,eva30M_shard0of2,eva30M_shard1of2,eva30M_merged}

run_shard() {
  local gpu="$1"
  local model_label="$2"
  local shard_idx="$3"
  local checkpoint="$4"
  local wt_cache="$5"
  local model_name="$6"
  local out_subdir="$7"
  local cname="eva_pos_${RUN_NAME}_${model_label}_s${shard_idx}"
  local log="${BASE_OUT}/logs/${model_name}.log"

  echo "[$(date '+%F %T')] START ${model_name} on GPU ${gpu}" | tee -a "${log}"
  docker run --rm --gpus "\"device=${gpu}\"" \
    -v "${HOST_EVA}:/eva" \
    -w /eva \
    --name "${cname}" \
    "${IMAGE}" \
    /composer-python/python "${SCRIPT}" \
      --checkpoint "${checkpoint}" \
      --dataset "${DATASET}" \
      --output-dir "${MOUNT_OUT}/${out_subdir}" \
      --model-name "${model_name}" \
      --wt-cache "${wt_cache}" \
      --device cuda:0 \
      --max-seqlen 8192 \
      --num-shards 2 \
      --shard-index "${shard_idx}" \
      --progress-every 100 \
    >> "${log}" 2>&1
  echo "[$(date '+%F %T')] DONE ${model_name}" | tee -a "${log}"
}

merge_model() {
  local merged_name="$1"
  local shard0="$2"
  local shard1="$3"
  local merged_dir="$4"
  local log="${BASE_OUT}/logs/${merged_name}_merge.log"

  echo "[$(date '+%F %T')] MERGE ${merged_name}" | tee -a "${log}"
  docker run --rm \
    -v "${HOST_EVA}:/eva" \
    -w /eva \
    "${IMAGE}" \
    /composer-python/python "${MERGE_SCRIPT}" \
      --input-dirs "${MOUNT_OUT}/${shard0}" "${MOUNT_OUT}/${shard1}" \
      --output-dir "${MOUNT_OUT}/${merged_dir}" \
      --merged-model-name "${merged_name}" \
    >> "${log}" 2>&1
  echo "[$(date '+%F %T')] MERGED ${merged_name}" | tee -a "${log}"
}

echo "Run name: ${RUN_NAME}"
echo "Output: ${BASE_OUT}"

run_shard 2 "1400M" 0 "/eva/checkpoint/clm" "/eva/data/position_ablation_eukaryote/delta_ll_results_1400M_8192.json" "eva1400M_lineage_s0of2" "eva1400M_shard0of2" &
pid_a=$!
run_shard 3 "1400M" 1 "/eva/checkpoint/clm" "/eva/data/position_ablation_eukaryote/delta_ll_results_1400M_8192.json" "eva1400M_lineage_s1of2" "eva1400M_shard1of2" &
pid_b=$!
wait "${pid_a}" "${pid_b}"

merge_model "eva1400M_lineage_eukaryote_position_ablation" "eva1400M_shard0of2" "eva1400M_shard1of2" "eva1400M_merged"

run_shard 2 "30M" 0 "/eva/checkpoint/30M_1124/checkpoint-56844" "/eva/data/position_ablation_eukaryote/delta_ll_results_30M_8192.json" "eva30M_lineage_s0of2" "eva30M_shard0of2" &
pid_c=$!
run_shard 3 "30M" 1 "/eva/checkpoint/30M_1124/checkpoint-56844" "/eva/data/position_ablation_eukaryote/delta_ll_results_30M_8192.json" "eva30M_lineage_s1of2" "eva30M_shard1of2" &
pid_d=$!
wait "${pid_c}" "${pid_d}"

merge_model "eva30M_lineage_eukaryote_position_ablation" "eva30M_shard0of2" "eva30M_shard1of2" "eva30M_merged"

echo "[$(date '+%F %T')] ALL DONE ${RUN_NAME}" | tee -a "${BASE_OUT}/logs/run.log"
