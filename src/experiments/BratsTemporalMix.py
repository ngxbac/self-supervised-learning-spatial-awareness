from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn

from catalyst.dl import ConfigExperiment
from augmentation import train_brats_aug, valid_brats_aug
from datasets import BratsTemporalMixDataset


class BratsTemporalMix(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)
        torch.cuda.manual_seed_all(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        return model_

    def get_datasets(
            self,
            stage: str,
            train_root_dir: str,
            valid_root_dir: str,
            train_csv: str,
            valid_csv: str,
            image_size = [224, 224],
    ):
        datasets = OrderedDict()

        transform = train_brats_aug(image_size)
        train_set = BratsTemporalMixDataset(
            root_dir=train_root_dir,
            csv_file=train_csv,
            transform=transform,
        )
        datasets["train"] = train_set

        transform = valid_brats_aug(image_size)
        valid_set = BratsTemporalMixDataset(
            root_dir=train_root_dir,
            csv_file=valid_csv,
            transform=transform,
        )
        datasets["valid"] = valid_set

        return datasets