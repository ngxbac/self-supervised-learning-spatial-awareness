#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export CUDA_VISIBLE_DEVICES=0
RUN_CONFIG=brats_temporalmix.yml


log_name=resnet34-tx-200
LOGDIR=/logs/proxy/brats/${log_name}/
USE_WANDB=${WANDB} catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --monitoring_params/name=${log_name}:str \
    --verbose