from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn

from catalyst.dl import ConfigExperiment
from augmentation import train_brats_aug, valid_brats_aug
from datasets import Brats2019Dataset


class BratsSegment(ConfigExperiment):
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
            train_root_dir: str,
            valid_root_dir: str,
            train_csv: str,
            valid_csv: str,
            image_size = [224, 224],
            data = "WT",
            **kwargs
    ):
        datasets = OrderedDict()
        transform = train_brats_aug(image_size)
        train_set = Brats2019Dataset(
            root_dir=train_root_dir,
            csv_file=train_csv,
            transform=transform,
            data=data
        )
        datasets["train"] = train_set

        transform = valid_brats_aug(image_size)
        valid_set = Brats2019Dataset(
            root_dir=train_root_dir,
            csv_file=valid_csv,
            transform=transform,
            data=data
        )
        datasets["valid"] = valid_set

        return datasets