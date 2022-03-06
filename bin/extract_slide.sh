#!/usr/bin/env bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# For StructSeg
#python src/preprocessing.py extract-2d      --root /data/Thoracic_OAR/ \
#                                            --save_dir /data/Thoracic_OAR_2d/


# For Brats
python src/preprocessing.py preprocess-brats19-train  --root /data/brats2019/training/ \
                                                      --save_dir /data/brats2019/training_preprocess_raw/


#python src/preprocessing.py preprocess-brats19-valid  --root /data/brats2019/validation/ \
#                                                      --save_dir /data/brats2019/validation_preprocess/
