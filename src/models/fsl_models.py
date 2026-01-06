"""
https://huggingface.co/MahmoodLab/UNI
https://github.com/mahmoodlab/UNI
https://www.nature.com/articles/s41591-024-02857-3.epdf?sharing_token=CzM2TCW_6hilYJ6BCMgx5dRgN0jAjWel9jnR3ZoTv0PwDLGHgijc_t52lQyxVvw552KDCUhMbS4KuO_rvLnP6S1BpmIj9puojkF8lfR5R8uEX08B0FxePgIH0t7DovKvZF4NHQKlq4TZHGAA1wEIdkYKvcr8nUsaa-nNYbNw3JI%3D
"""
import os
from typing import Optional, Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import pathology_foundation_models as pfm

local_dir = './assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/'

class WrappedFsl(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module | str, 
        hidden_dim: Optional[int] = None, 
        embedding_dim: Optional[int] = None,
        hf_token: Optional[str] = None,
        device: str = 'cuda',
    ):
        super().__init__()

        self.hidden_dim = hidden_dim if hidden_dim else 512
        self.embedding_dim = embedding_dim if embedding_dim else 128
        self.device = device

        if isinstance(backbone, str):
            backbone = pfm.models.load_foundation_model(backbone, token=hf_token, device=self.device)

        assert isinstance(backbone, nn.Module)
        self.add_module('backbone', backbone)

        self.backbone.eval()
        with torch.no_grad():
            test_tensor = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.float32).to(self.device)
            self.backbone_out_dim = self.backbone(test_tensor).shape[-1]

        # Freeze backbone if needed
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Create projection
        self.add_module('projection', nn.Sequential(
            nn.Linear(self.backbone_out_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        ).to(self.device))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval() # assert that the backbone always is in evaluation mode
        return self

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
        return torch.stack([proto_neg, proto_pos], dim=1)  # (2, D)

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
        with torch.no_grad():
            x = self.backbone(x)

        x = x.clone() # required by autograd.
        x = self.projection(x)
        return x

    @staticmethod
    def from_config(model_config: dict[str, Any], hf_token: Optional[str] = None):
        return WrappedFsl(
            backbone=model_config['model_name'], 
            hidden_dim=model_config.get('hidden_dim'),
            embedding_dim=model_config.get('embedding_dim'),
            hf_token=hf_token,
        )

if __name__ == '__main__':
    import os
    model = WrappedFsl('uni', hf_token=os.getenv("HF_TOKEN"), device='cuda')
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
