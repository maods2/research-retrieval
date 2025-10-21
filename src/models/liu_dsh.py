import torch
from torch import nn
import torch.nn.functional as F
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




class LiuDSH(nn.Module):
    """
    Liu Deep Supervised Hashing (DSH) model with configurable backbone.
    
    This model implements the deep supervised hashing approach for learning
    binary hash codes for image retrieval tasks. It can use different backbones
    (ResNet, DINO, UNI, etc.) for feature extraction.
    """
    
    def __init__(self, model_config: dict):
        super().__init__()
        
        # Extract backbone configuration
        backbone_config = model_config.get('backbone', {'type': 'resnet'})
        self.backbone = get_backbone_model(backbone_config)
        
        backbone_output_dim = self._get_backbone_output_dim(backbone_config)

        
        # Hash code size
        code_size = model_config.get('code_size', 32)
        
        # Create hash layer
        self.hash_layer = nn.Linear(
            in_features=backbone_output_dim, 
            out_features=code_size
        )
    
    def _get_backbone_output_dim(self, backbone, input_shape=(1, 3, 224, 224)):
        """Get backbone output dimension using a dummy forward pass"""
        with torch.no_grad():
            dummy_output = backbone(torch.randn(*input_shape))
        return dummy_output.shape[1]

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Generate hash codes
        hash_codes = self.hash_layer(features)
        
        return hash_codes
