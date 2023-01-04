from typing import Optional

import timm
import torch
import torchvision.models as models
from torch import nn

from src.config import net_config, torch_config


def define_net(
    model_name: str = "resnet18",
    freeze_grads: bool = False,
    outputs: int = net_config.outputs,
    pretrained: bool = False,
    weights: Optional[str] = None,
):
    if model_name == "resnet18":
        pretrained_weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=pretrained_weights)
    elif model_name == "resnet50":
        pretrained_weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=pretrained_weights)
    elif model_name == "resnest-50":
        # small fix for "urllib.error.HTTPError: HTTP Error 403: rate limit exceeded" bug
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        torch.hub.list("zhanghang1989/ResNeSt", force_reload=True)
        model = torch.hub.load(
            "zhanghang1989/ResNeSt", "resnest50", pretrained=pretrained
        )
    elif model_name == "rexnet-100":
        model = timm.create_model("rexnet_100", pretrained=pretrained)
    elif model_name == "convnext":
        # convnext_tiny_in22k
        # convnext_tiny_in22ft1k
        # convnext_tiny - 1k ema
        model = timm.create_model("convnext_tiny_in22k", pretrained=pretrained)
    else:
        raise Exception(
            f"Unsupported model_name, expected resnet18, resnet50, resnest-50, rexnet-100, convnext got {model_name}"
        )

    if freeze_grads:
        for params in model.parameters():
            params.requires_grad_ = False

    try:
        model.head.fc = nn.Linear(
            in_features=model.head.fc.in_features, out_features=outputs, bias=True
        )
    except AttributeError:
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=outputs, bias=True
        )

    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu").state_dict())

    model.to(torch_config.device)

    return model
