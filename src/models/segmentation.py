from segmentation_models_pytorch.unet.model import *
from segmentation_models_pytorch.encoders import encoders

import torch
import torch.nn as nn


def modified_get_encoder(name, in_channels=3, depth=5, weights=None):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    encoder.set_in_channels(in_channels)

    if weights is not None:
        # settings = encoders[name]["pretrained_settings"][weights]
        state_dict = torch.load(weights)["model_state_dict"]
        encoder.load_state_dict(state_dict)
        print("Loaded ", weights)

    return encoder


class SSUnet(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights = None,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            aux_params: Optional[dict] = None,
    ):
        super(SSUnet, self).__init__()

        # import pdb; pdb.set_trace()

        self.encoder = modified_get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.initialize()