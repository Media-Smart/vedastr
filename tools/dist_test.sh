#!/usr/bin/env bash
if (($# < 3)); then
    echo "Uasage: bash tools/dist_train.sh config_file checkpoint gpu_nums"
    exit 1
fi

CONFIG="$1"
CHECKPOINT="$2"
GPUS="$3"
PORT="$((29400 + RANDOM % 100))"

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --distribute ${@:4}
