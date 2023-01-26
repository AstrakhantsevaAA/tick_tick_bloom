import torch.nn as nn
from torch import cat

from src.config import data_config


class HrrrNet(nn.Module):
    def __init__(self, feature_extractor, outputs: int = 1):
        super().__init__()
        self.cnn = feature_extractor

        try:
            in_features = self.cnn.head.fc.in_features
        except AttributeError:
            in_features = self.cnn.fc.in_features

        self.cnn.reset_classifier(0)
        self.fc1 = nn.Linear(in_features + len(data_config.best_features), 1000)
        self.fc2 = nn.Linear(1000, outputs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, hrrr):
        x1 = self.cnn(image)
        x2 = hrrr

        x = cat((x1, x2), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
