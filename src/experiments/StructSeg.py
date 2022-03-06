from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn

from catalyst.dl import ConfigExperiment
from augmentation import train_seg_aug, valid_seg_aug
from datasets import StructSegTrain2D


class StructSeg(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):
        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        # if stage in ["debug", "stage1"]:
        #     for param in model_.encoder.parameters():
        #         param.requires_grad = False
        # elif stage == "stage2":
        #     for param in model_.encoder.parameters():
        #         param.requires_grad = True

        return model_

    def get_datasets(
            self,
            stage: str,
            train_csv: str,
            valid_csv: str,
            image_size = [224, 224],
            **kwargs
    ):
        print(train_csv)
        print(valid_csv)
        datasets = OrderedDict()
        transform = train_seg_aug(image_size)
        train_set = StructSegTrain2D(
            csv_file=train_csv,
            transform=transform,
        )
        datasets["train"] = train_set

        transform = valid_seg_aug(image_size)
        valid_set = StructSegTrain2D(
            csv_file=valid_csv,
            transform=transform,
        )
        datasets["valid"] = valid_set

        return datasets