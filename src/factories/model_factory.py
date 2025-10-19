import os
import sys
from typing import Optional, Any

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

import pathology_foundation_models as pfm

# FSL
from models.fsl_models import WrappedFsl

from models.dino import DINO
from models.dino import DINOv2
from models.resnet import ResNet
from models.vit import ViT
from models.n_branch_mlp import N_BranchMLP
from utils.checkpoint_utils import load_checkpoint


def get_model(model_config: dict[str, Any], hf_token: Optional[str] = None):
    assert 'model_code' in model_config.keys(), "No `model_code` key found. Cannot construct model."
    model_code = model_config['model_code'].lower().strip()

    if pfm.models.is_model_available(model_str=model_code):
        model = pfm.models.load_foundation_model(model_type=model_code, token=hf_token)

    elif model_code == 'resnet':
        model = ResNet(model_config)

    elif model_code == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_code == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])

    elif model_code == 'vit':
        model = ViT(model_name=model_config['model_name'])

    ################### Few-Shot Learning Models ######################################

    elif 'fsl' in model_code:
        model = WrappedFsl.from_config(model_config=model_config, hf_token=hf_token)

    ################### n-Branch MLP Attention Metric #################################

    elif 'branch_mlp' in model_code:
        model = N_BranchMLP(model_config)
        print(model)

    else:
        raise ValueError(f'Model {model_code} is not supported')

    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model
