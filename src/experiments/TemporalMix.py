from collections import OrderedDict

import numpy as np
import random

import torch
import torch.nn as nn

from catalyst.dl import ConfigExperiment
from augmentation import train_aug, valid_aug
from datasets import TemporalMixDataset


class TemporalMix(ConfigExperiment):
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
            train_file: str,
            valid_file: str,
            image_size = [224, 224],
    ):
        datasets = OrderedDict()

        train_set = TemporalMixDataset(
            csv_file=train_file,
            transform=train_aug(image_size),
        )

        valid_set = TemporalMixDataset(
            csv_file=valid_file,
            transform=valid_aug(image_size),
        )
        # from torch.utils.data import ConcatDataset
        # concat_set = ConcatDataset([train_set, valid_set])
        datasets["train"] = train_set
        datasets["valid"] = valid_set

        return datasets