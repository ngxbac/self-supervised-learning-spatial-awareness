from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from catalyst.dl import ConfigExperiment
from augmentation import train_aug, valid_aug
from datasets import CHAOSDataset


class CHAOS(ConfigExperiment):
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

        return model_

    def get_datasets(
            self,
            stage: str,
            train_ct_dir: str,
            train_pred_dir: str,
            valid_ct_dir: str,
            valid_pred_dir: str,
            all_data: bool = False,
            image_size = [224, 224],
            **kwargs
    ):
        datasets = OrderedDict()
        train_set = CHAOSDataset(
            ct_dir=train_ct_dir,
            pred_dir=train_pred_dir,
            transform=train_aug(image_size),
        )

        valid_set = CHAOSDataset(
            ct_dir=valid_ct_dir,
            pred_dir=valid_pred_dir,
            transform=valid_aug(image_size),
        )

        if all_data:
            concat_dataset = ConcatDataset(
                [train_set, valid_set]
            )
            datasets["train"] = concat_dataset
            datasets["valid"] = concat_dataset
        else:
            datasets["train"] = train_set
            datasets["valid"] = valid_set

        return datasets