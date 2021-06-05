#!/usr/bin/env bash

if (($# < 3)); then
  echo "Uasage: bash tools/dist_test.sh config_file checkpoint gpus_to_use"
  exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"

IFS=', ' read -r -a gpus <<<"${GPUS}"
NGPUS="${#gpus[@]}"
PORT="$((29400 + RANDOM % 100))"

export CUDA_VISIBLE_DEVICES=${GPUS}

PYTHONPATH="$(dirname "$0")/..":${PYTHONPATH} \
    python -m torch.distributed.launch \
        --nproc_per_node="${NGPUS}" \
        --master_port=${PORT} \
        "$(dirname "$0")"/test.py \
            "$CONFIG" \
            "$CHECKPOINT" \
            --distribute \
            "${@:4}"
