import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm import create_model
from timm.models.layers.activations import Swish
from timm.models.layers import SelectAdaptivePool2d
from torchvision.models.inception import InceptionA


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class TIMMModels(nn.Module):
    def __init__(self, model_name, num_classes):
        super(TIMMModels, self).__init__()
        self.model = create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=3,
        )

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class TIMMetricLearningMModels(nn.Module):
    def __init__(self, model_name, num_classes, aux=False):
        super(TIMMetricLearningMModels, self).__init__()
        self.model = create_model(
            model_name=model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3,
        )

        # Deep-supervised learning
        self.aux = aux
        if self.aux:
            self.num_p2_features = self.model.layer2[-1].bn2.num_features
            self.num_p3_features = self.model.layer3[-1].bn2.num_features

            self.ds1 = nn.Sequential(
                # Fire(self.num_p2_features, 16, 64, 64),
                # Fire(128, 16, 64, 64),
                # Fire(128, 32, 128, 128),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire(256, 32, 128, 128),
                # Fire(256, 48, 192, 192),
                # Fire(384, 48, 192, 192),
                # Fire(384, 64, 256, 256),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire(512, 64, 256, 256),

                nn.Conv2d(
                    self.num_p2_features, self.num_p2_features * 4, kernel_size=(1, 1), stride=(1, 1), bias=False
                ),
                nn.BatchNorm2d(self.num_p2_features * 4),
                Swish(),
                SelectAdaptivePool2d(),
                Flatten(),
                nn.Linear(self.num_p2_features * 4, num_classes)
            )

            self.ds2 = nn.Sequential(
                # Fire(self.num_p3_features, 16, 64, 64),
                # Fire(128, 16, 64, 64),
                # Fire(128, 32, 128, 128),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire(256, 32, 128, 128),
                # Fire(256, 48, 192, 192),
                # Fire(384, 48, 192, 192),
                # Fire(384, 64, 256, 256),
                # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # Fire(512, 64, 256, 256),
                nn.Conv2d(
                    self.num_p3_features, self.num_p3_features * 4, kernel_size=(1, 1), stride=(1, 1), bias=False
                ),
                nn.BatchNorm2d(self.num_p3_features * 4),
                Swish(),
                SelectAdaptivePool2d(),
                Flatten(),
                nn.Linear(self.num_p3_features * 4, num_classes)
            )

        # Arcface
        features_num = self.model.num_features
        embedding_size = 512

        self.neck = nn.Sequential(
            nn.BatchNorm1d(features_num),
            nn.Linear(features_num, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        )
        self.arc_margin_product = ArcMarginProduct(embedding_size, num_classes)
        self.arc_loss = ArcFaceLoss()
        self.head_arcface = nn.Linear(embedding_size, num_classes)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def embed(self, x):
        if self.aux:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x_ds1 = x
            x = self.model.layer3(x)
            x_ds2 = x
            x = self.model.layer4(x)
        else:
            x = self.model.forward_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        embedding = self.neck(x)

        if self.aux:
            return embedding, x_ds1, x_ds2
        else:
            return embedding

    def metric_classify(self, embedding):
        return self.arc_margin_product(embedding)

    def classify(self, embedding):
        return self.head_arcface(embedding)

    def forward(self, x):
        if self.aux:
            embedding, x_ds1, x_ds2 = self.embed(x)
            logits = self.classify(embedding)
            logits_ml = self.metric_classify(embedding)

            # import pdb; pdb.set_trace()

            logits_ds1 = self.ds1(x_ds1)
            logits_ds2 = self.ds2(x_ds2)
            return logits, logits_ml, logits_ds1, logits_ds2, self.arc_loss
        else:
            embedding = self.embed(x)
            logits = self.classify(embedding)
            logits_ml = self.metric_classify(embedding)
            return logits, logits_ml, self.arc_loss