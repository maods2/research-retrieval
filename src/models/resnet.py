from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet18 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, config):
        super(ResNet34, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet34 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, config):
        super(ResNet50, self).__init__()
        freeze_params = config.get('freeze_params', True)
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = (
            nn.Identity()
        )  # Remove the final fully connected layer

        if freeze_params:
            print('Freezing ResNet50 parameters')
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        return x


class ResNet(nn.Module):
    def __init__(self, model_config):
        super(ResNet, self).__init__()
        model_name = model_config.get('model_name', 'resnet18').lower()
        if model_name == 'resnet18':
            self.model = ResNet18(model_config)
        elif model_name == 'resnet34':
            self.model = ResNet34(model_config)
        elif model_name == 'resnet50':
            self.model = ResNet50(model_config)
        else:
            raise ValueError(f'Unknown model_name: {model_name}')

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)  # Example input tensor
    model = ResNet18()
    output = model(x)
    print(f'ResNet18 output shape: {output.shape}')
    model = ResNet34()
    output = model(x)
    print(f'ResNet34 output shape: {output.shape}')
    model = ResNet50()
    output = model(x)
    print(f'ResNet50 output shape: {output.shape}')

# ResNet18 output shape: torch.Size([1, 512])
# ResNet34 output shape: torch.Size([1, 512])
# ResNet50 output shape: torch.Size([1, 2048])
