import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

import pathology_foundation_models as pfm

from src.models.fsl_models import DinoFsl
from src.models.fsl_models import DINOv2Fsl
from src.models.fsl_models import PhikonFsl
from src.models.fsl_models import ResNetFsl
from src.models.fsl_models import UNIFsl
from src.models.fsl_models import Virchow2Fsl
from src.models.fsl_models import ViTFsl
from src.models.dino import DINO
from src.models.dino import DINOv2
from src.models.resnet import ResNet
from src.models.vit import ViT
from src.utils.checkpoint_utils import load_checkpoint


def get_model(model_config):
    model_code = model_config.get('model_code').lower()

    # HACK: Transition solution
    if model_code.upper() in pfm.models.FoundationModelEnum.__members__.keys():
        model = pfm.models.load_foundation_model(model_type=model_code)

    elif model_code == 'resnet':
        model = ResNet(model_config)

    elif model_code == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_code == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])

    elif model_code == 'vit':
        model = ViT(model_name=model_config['model_name'])

    ################### Few-Shot Learning Models ######################################

    elif model_code == 'resnet_fsl':   # Pathology Foundation Model
        model = ResNetFsl(model_config)

    elif model_code == 'dino_fsl':
        model = DinoFsl(model_config)

    elif model_code == 'dinov2_fsl':
        model = DINOv2Fsl(model_config)

    elif model_code == 'vit_fsl':
        model = ViTFsl(model_config)

    elif model_code == 'uni_fsl':   # Pathology Foundation Model
        model = UNIFsl(model_config)

    elif model_code == 'UNI2-h_fsl':   # Pathology Foundation Model
        # UNI2-h is a variant of UNI, so we can use the same class,
        # but we need to ensure the model_config is correctly set
        model = UNIFsl(model_config)

    elif model_code == 'virchow2_fsl':   # Pathology Foundation Model
        model = Virchow2Fsl(model_config)

    elif model_code == 'phikon_fsl':   # Pathology Foundation Model
        model = PhikonFsl(model_config)

    elif model_code == 'phikon-v2_fsl':   # Pathology Foundation Model
        # Phikon-v2 is a variant of Phikon, so we can use the same class,
        # but we need to ensure the model_config is correctly set
        model = PhikonFsl(model_config)

    else:
        raise ValueError(f'Model {model_code} is not supported')

    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model
