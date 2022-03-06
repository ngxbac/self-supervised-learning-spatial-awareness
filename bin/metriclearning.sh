#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


export CUDA_VISIBLE_DEVICES=0
export USE_WANDB=0
export USE_ALCHEMY=0
export USE_NEPTUNE=0

# Config experiments
export TRAIN_HERBARIUM=0
export TRAIN_PLANT=1

#export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjY1MGUxYTQtNGU5Ni00NzU4LWEwMjctNjgxMGU2ZWE1ZWJiIn0="
RUN_CONFIG=metriclearning.yml


for model_name in resnet18; do
    for fold in 0; do
        log_name=${model_name}-moreaug-$fold
        LOGDIR=/logs/metriclearning/${log_name}/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --model_params/model_name=${model_name}:str \
            --stages/data_params/fold=${fold}:int \
            --verbose
    done
done