import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from catalyst.utils import get_activation_fn
# from catalyst.contrib.nn.criterion import dice
from catalyst.utils import metrics


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MetricLearningLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, ratio=0.5, aux=False, num_classes=4):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(MetricLearningLoss, self).__init__()
        self.ratio = ratio
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.aux = aux
        self.num_classes = num_classes

    def _forward_metriclearning(self, logits, logits_ml, loss_fn, targets):
        ohe = F.one_hot(targets, self.num_classes)
        classification_loss = self.classification_loss_fn(logits, targets)
        arcface_loss = loss_fn(logits_ml, ohe)
        return self.ratio * classification_loss + (1 - self.ratio) * arcface_loss

    def _forward_deepmetriclearning(self, logits, logits_ml, logits_ds1, logits_ds2, loss_fn, targets):
        ohe = F.one_hot(targets, self.num_classes)
        classification_loss = self.classification_loss_fn(logits, targets)
        arcface_loss = loss_fn(logits_ml, ohe)
        ds_loss1 = self.classification_loss_fn(logits_ds1, targets)
        ds_loss2 = self.classification_loss_fn(logits_ds2, targets)
        return (classification_loss + arcface_loss + ds_loss1 + ds_loss2) / 4

    def forward(self, **kwargs):
        logits = kwargs.get("logits", None)
        logits_ml = kwargs.get("logits_ml", None)
        logits_ds1 = kwargs.get("logits_ds1", None)
        logits_ds2 = kwargs.get("logits_ds2", None)
        loss_fn = kwargs.get("loss_fn", None)
        targets = kwargs.get("targets", None)

        if self.aux:
            return self._forward_deepmetriclearning(logits, logits_ml, logits_ds1, logits_ds2, loss_fn, targets)
        else:
            return self._forward_metriclearning(logits, logits_ml, loss_fn, targets)


def dice__(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = None
):
    """
    Computes the dice metric

    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


class MultiDiceLoss(nn.Module):
    def __init__(
        self,
        activation: str = "Softmax",
        num_classes: int = 7,
        weight = None,
        dice_weight: float = 0.3,
    ):
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(np.asarray(weight).astype(np.float32))
        else:
            weight = None
        self.num_classes = num_classes
        self.activation = activation
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        activation_fnc = get_activation_fn(self.activation)
        logits_softmax = activation_fnc(logits)

        ce_loss = self.ce_loss(logits, targets)

        dice_loss = 0
        for cls in range(self.num_classes):
            targets_cls = (targets == cls).float()
            outputs_cls = logits_softmax[:, cls]
            score = 1 - metrics.dice(outputs_cls, targets_cls, eps=1e-7, activation='none', threshold=None)
            dice_loss += score / self.num_classes

        loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return loss