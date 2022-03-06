#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


export CUDA_VISIBLE_DEVICES=0
export USE_WANDB=0
export USE_ALCHEMY=0
export USE_NEPTUNE=0

RUN_CONFIG=structseg_segment.yml


for model_name in resnet34; do
    for fold in 1 2 3 4; do
        nvidia-smi
        log_name=${model_name}-standard-$fold
        LOGDIR=/logs/papers/structseg2/${log_name}/
        train_csv=./csv/5folds/train_$fold.csv
        valid_csv=./csv/5folds/valid_$fold.csv
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/train_csv=${train_csv}:str \
            --stages/data_params/valid_csv=${valid_csv}:str \
            --verbose
    done
done