import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, vgg11


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()

    return model


def set_in_channels(model, in_channels):
    """Change first convolution chennels"""
    if in_channels == 3:
        return model

    return patch_first_conv(model=model, in_channels=in_channels)


def proxy_model(num_classes=5, in_channels=3):
    model = resnet34(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=in_features, out_features=num_classes
    )

    model = set_in_channels(model, in_channels)

    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    return model


