import torch.nn as nn
from torchvision import models

from src import config


def get_model(freeze_backbone=False):
    """
    ResNet50 backbone, pretrained on ImageNet, with a new classification head
    for 5 cassava disease classes.

    freeze_backbone=True trains only the new head (fast, good starting point).
    freeze_backbone=False fine-tunes the whole network (slower, usually more accurate).
    """
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        for param in net.parameters():
            param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, config.NUM_CLASSES),
    )

    return net.to(config.DEVICE)
