import torch.nn as nn
from torch import cat

from src.config import data_config


class HrrrNet(nn.Module):
    def __init__(self, feature_extractor, outputs: int = 1, meta: bool = False):
        super().__init__()
        self.cnn = feature_extractor
        self.meta = meta
        if meta:
            layer = self.cnn.features[0].conv_dw.conv
            self.cnn.features[0].conv_dw.conv = nn.Conv2d(
                in_channels=layer.in_channels
                + len(data_config.meta_keys)
                + data_config.num_scl_classes,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias,
            )

        try:
            in_features = self.cnn.head.fc.in_features
        except AttributeError:
            in_features = self.cnn.fc.in_features

        self.cnn.reset_classifier(0)
        self.fc1 = nn.Linear(in_features + len(data_config.best_features), 1000)
        self.fc2 = nn.Linear(1000, outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward_features(self, x, meta=None):
        x = self.cnn.stem(x)
        if meta is not None:
            x = cat((x, meta), dim=1)
        x = self.cnn.features(x)
        return x

    def forward(self, image, hrrr, meta=None):
        x1 = self.forward_features(image, meta)
        x1 = self.cnn.head(x1)
        x2 = hrrr
        x = cat((x1, x2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
