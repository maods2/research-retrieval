import torch
from torch import nn
import sys
import os

# Add the src directory to the path to import other models
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.resnet import ResNet
from src.models.dino import DINO, DINOv2
from src.models.uni import UNI


def get_backbone_model(backbone_config):
    """
    Factory function to create backbone models.
    
    Args:
        backbone_config (dict): Configuration for the backbone model
        
    Returns:
        nn.Module: The backbone model
    """
    backbone_type = backbone_config.get('type', 'resnet').lower()
    
    if backbone_type == 'resnet':
        return ResNet(backbone_config)
    elif backbone_type == 'dino':
        return DINO(model_name=backbone_config.get('model_name', 'vit_small_patch16_224_dino'))
    elif backbone_type == 'dinov2':
        return DINOv2(model_name=backbone_config.get('model_name', 'dinov2_vitl14'))
    elif backbone_type == 'uni':
        return UNI(model_name=backbone_config.get('model_name', 'uni'))
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}")


class ProjectionHead(nn.Module):
    def __init__(self, base_model, hidden_dim=512, out_dim=128):
        super().__init__()
        self.backbone = base_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.proj(features)


class SupCon(nn.Module):
    """
    Supervised Contrastive Learning model with configurable backbone.
    
    This model can use different backbones (ResNet, DINO, UNI, etc.) for 
    supervised contrastive learning tasks.
    """
    
    def __init__(self, model_config: dict):
        super().__init__()
        
        # Extract backbone configuration
        backbone_config = model_config.get('backbone', {'type': 'resnet'})
        self.backbone = get_backbone_model(backbone_config)
        
        # Get backbone output dimensions
        backbone_output_dim = self._get_backbone_output_dim()
        
        # Projection head configuration
        hidden_dim = model_config.get('hidden_dim', 512)
        out_dim = model_config.get('out_dim', 128)
        
        # Create projection head
        self.projection_head = ProjectionHead(
            base_model=self.backbone,
            hidden_dim=backbone_output_dim,
            out_dim=out_dim
        )
    
    def _get_backbone_output_dim(self, input_shape=(1, 3, 224, 224)):
        """Get backbone output dimension using a dummy forward pass"""
        with torch.no_grad():
            dummy_output = self.backbone(torch.randn(*input_shape))
        return dummy_output.shape[1]
    
    def forward(self, x):
        return self.projection_head(x)
