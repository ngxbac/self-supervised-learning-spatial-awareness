#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export CUDA_VISIBLE_DEVICES=0
RUN_CONFIG=structseg_temporalmix.yml


log_name=resnet34-tx-500
LOGDIR=/logs/papers/structseg2/${log_name}/
USE_WANDB=${WANDB} catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --monitoring_params/name=${log_name}:str \
    --verbose