# flake8: noqa
import os
from catalyst.dl import registry
from catalyst.contrib.models.cv import segmentation as m
import experiments as exp

if os.environ.get("USE_WANDB", "0") == "1":
    from catalyst.dl import SupervisedWandbRunner as Runner
elif os.environ.get("USE_NEPTUNE", "0") == "1":
    from catalyst.dl import SupervisedNeptuneRunner as Runner
elif os.environ.get("USE_ALCHEMY", "0") == "1":
    from catalyst.dl import SupervisedAlchemyRunner as Runner
else:
    from catalyst.dl import SupervisedRunner as Runner

from .models import (
    TIMMModels, TIMMetricLearningMModels, proxy_model, SSUnet
)
from .callbacks import (
    MultiDiceCallback, MacroF2ScoreCallback
)

from .losses import(
    LabelSmoothingCrossEntropy, MetricLearningLoss, MultiDiceLoss
)

registry.MODELS.add_from_module(m)
registry.EXPERIMENTS.add_from_module(exp)

registry.Model(TIMMModels)
registry.Model(TIMMetricLearningMModels)
registry.Model(proxy_model)
registry.Model(SSUnet)

registry.Callback(MultiDiceCallback)
registry.Callback(MacroF2ScoreCallback)

registry.Criterion(LabelSmoothingCrossEntropy)
registry.Criterion(MetricLearningLoss)
registry.Criterion(MultiDiceLoss)