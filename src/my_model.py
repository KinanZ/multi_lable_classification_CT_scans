from torchvision import models as models
import torch.nn as nn


def resnet_model(num_classes, pretrained, requires_grad=True):
    model = models.resnet18(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    return model
