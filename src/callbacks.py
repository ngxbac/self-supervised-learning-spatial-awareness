from typing import Any, List, Optional, Union  # isort:skip
# import logging

from catalyst.dl import Callback, CallbackOrder, MetricCallback
from catalyst.utils import get_activation_fn
# from catalyst.core import State

import torch
import torch.nn.functional as F


def _dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
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
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    # dice = 2 * (intersection + eps * (union == 0)) / (union + eps)
    dice = (2 * intersection + eps * (union == 0)) / (union + eps)

    return dice


class MultiDiceCallback(Callback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        activation: str = "Softmax",
        num_classes : int = 7,
        class_names = None,
        include_bg = False,
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.activation = activation
        self.num_classes = num_classes
        self.class_names = class_names
        self.include_bg = include_bg

    def on_batch_end(self, state):
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]

        activation_fnc = get_activation_fn(self.activation)
        outputs = activation_fnc(outputs)
        _, outputs = outputs.max(dim=1)

        dice = 0
        start_idx = 0 if self.include_bg else 1
        for cls in range(start_idx, self.num_classes):
            targets_cls = (targets == cls).float()
            outputs_cls = (outputs == cls).float()
            score = _dice(outputs_cls, targets_cls, eps=1e-7, activation='none', threshold=None)
            if self.class_names is not None:
                state.batch_metrics[f"{self.prefix}_{self.class_names[cls]}"] = score
            if self.include_bg:
                dice += score / self.num_classes
            else:
                dice += score / (self.num_classes - 1)
        state.batch_metrics[self.prefix] = dice


from sklearn.metrics import fbeta_score
def macro_f2_score(
    outputs,
    targets,
):
    return fbeta_score(targets, outputs, beta=2, average='samples')


class MacroF2ScoreCallback(MetricCallback):
    """
    F1 score metric callback.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f2_score",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
        """
        super().__init__(
            prefix=prefix,
            metric_fn=macro_f2_score,
            input_key=input_key,
            output_key=output_key,
        )

    def on_loader_start(self, state):
        self.targets = []
        self.predicts = []

    def on_batch_end(self, state):
        outputs = self._get_output(state.batch_out, self.output_key)
        targets = self._get_input(state.batch_in, self.input_key)

        outputs = F.sigmoid(outputs)

        outputs = outputs.detach().cpu().numpy()
        outputs = outputs > 0.5
        targets = targets.detach().cpu().numpy()

        self.targets += list(targets)
        self.predicts += list(outputs)

    def on_loader_end(self, state):
        """
        Computes the metric and add it to epoch metrics
        """
        metric = self.metric_fn(self.predicts, self.targets) * self.multiplier
        state.loader_metrics[self.prefix] = metric
