from typing import Any, Optional

import timm
import torch
from torch import nn

from src.config import net_config, torch_config
from src.nets.hrrr_net import HrrrNet


def set_new_in_channels(model, new_in_channels: int, pretrained: bool = False):
    try:
        layer = model.cnn.stem.conv
    except AttributeError:
        layer = model.cnn.conv1

    new_layer = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias,
    )
    if pretrained:
        copy_weights = 0  # Here will initialize the weights from new channel with the red channel weights

        # Copying the weights from the old to the new layer
        new_layer.weight[:, : layer.in_channels, :, :].data = layer.weight.data.clone()

        # Copying the weights of the `copy_weights` channel of the old layer to the extra channels of the new layer
        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel : channel + 1, :, :].data = layer.weight[
                :, copy_weights : copy_weights + 1, ::
            ].data.clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

    try:
        model.cnn.stem.conv = new_layer
    except AttributeError:
        model.cnn.conv1 = new_layer

    return model


def parse_model_name(
    model_name: str, outputs: int, pretrained: bool = False, hrrr: bool = False
) -> Any:
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
        )
        if hrrr:
            model = HrrrNet(feature_extractor=model, outputs=outputs)
        else:
            model.cnn.head.fc = nn.Linear(
                in_features=model.head.fc.in_features,
                out_features=outputs,
            )
        return model
    except Exception:
        raise Exception(
            f"Unsupported model_name: {model_name}. \
            Find the correct model name there https://huggingface.co/docs/timm/main/models"
        )


def define_net(
    model_name: str = "resnet18",
    hrrr: bool = False,
    outputs: int = net_config.outputs,
    pretrained: bool = False,
    weights_resume: Optional[str] = None,
    new_in_channels: int = net_config.in_channels,
):
    model = parse_model_name(model_name, outputs, pretrained, hrrr)

    if new_in_channels:
        model = set_new_in_channels(model, new_in_channels, pretrained)

    if weights_resume:
        model.load_state_dict(
            torch.load(weights_resume, map_location="cpu").state_dict()
        )

    model.to(torch_config.device)

    return model
