"""
https://huggingface.co/MahmoodLab/UNI
https://github.com/mahmoodlab/UNI
https://www.nature.com/articles/s41591-024-02857-3.epdf?sharing_token=CzM2TCW_6hilYJ6BCMgx5dRgN0jAjWel9jnR3ZoTv0PwDLGHgijc_t52lQyxVvw552KDCUhMbS4KuO_rvLnP6S1BpmIj9puojkF8lfR5R8uEX08B0FxePgIH0t7DovKvZF4NHQKlq4TZHGAA1wEIdkYKvcr8nUsaa-nNYbNw3JI%3D
"""

from src.models.dino import DINO, DINOv2
from src.models.phikon import Phikon
from src.models.resnet import ResNet
from src.models.uni import UNI
from src.models.virchow2 import Virchow2
from src.models.vit import ViT
from torch import Tensor

import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


local_dir = './assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/'


class WrappedFsl(nn.Module):
    def __init__(self, backbone, hidden_dim=512, embedding_dim=128):
        super().__init__()
        self.backbone = backbone

        with torch.no_grad():
            test_tensor = torch.randn(1, 3, 224, 224)
            out_dim = self.backbone(test_tensor).shape[-1]
        # Create projection
        self.projection = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Freeze backbone if needed
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def compute_prototypes(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        prototypes = []
        for c in torch.unique(labels):
            class_mask = labels == c
            class_proto = embeddings[class_mask].mean(0)
            prototypes.append(class_proto)
        return torch.stack(prototypes)

    def predict_with_prototypes(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        dists = torch.cdist(query_embeddings, prototypes)
        return torch.argmin(dists, dim=1)

    def predict_probabilities(
        self, query_embeddings: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        dists = torch.cdist(query_embeddings, prototypes)
        return (-dists).softmax(dim=1)

    def compute_binary_prototypes(self, embeddings, labels, positive_label):
        # embeddings: (S, D) support embeddings
        # labels:    (S,) original labels, where positive_label is your "1" class
        pos_mask = labels == positive_label
        neg_mask = labels != positive_label

        proto_pos = embeddings[pos_mask].mean(0)    # (D,)
        proto_neg = embeddings[neg_mask].mean(0)    # (D,)

        # stack into [neg, pos]
        return torch.stack([proto_neg, proto_pos], dim=0)  # (2, D)

    def predict_binary(self, query_embeddings, prototypes):
        # query_embeddings: (Q, D), prototypes: (2, D)
        dists = torch.cdist(query_embeddings, prototypes)  # (Q, 2)
        # argmin → 0=neg, 1=pos
        return torch.argmin(dists, dim=1)

    def predict_binary_probabilities(self, query_embeddings, prototypes):
        dists = torch.cdist(query_embeddings, prototypes)  # (Q, 2)
        # argmin → 0=neg, 1=pos
        return (-dists).softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x

class ResNetFsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            ResNet(model_config),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

class DinoFsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            DINO(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

class DINOv2Fsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            DINOv2(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

class ViTFsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            ViT(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

class UNIFsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            UNI(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

class Virchow2Fsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
             Virchow2(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )
        
class PhikonFsl(WrappedFsl):
    def __init__(self, model_config):
        super().__init__(
            Phikon(model_name=model_config['model_name']),
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

   
        
if __name__ == '__main__':
    model = UNIFsl()
    model = model.to('cuda')
    model.eval()
    prototypes = torch.randn(5, 128).to('cuda')  # Dummy prototypes
    with torch.no_grad():
        # Dummy input tensor
        x = torch.randn(1, 3, 224, 224).to('cuda')
        output = model(x)
        prob = model.predict_probabilities(output, prototypes)
        print(output.shape)
        print(output)
        print(prob.shape)
        print(prob)
