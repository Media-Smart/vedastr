#!/usr/bin/env bash

if (($# < 2)); then
    echo "Uasage: bash tools/dist_train.sh config_file gpu_nums"
    exit 1
fi

CONFIG=$1
GPUS=$2
PORT="$((29400 + RANDOM % 100))"

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --distribute ${@:3}
