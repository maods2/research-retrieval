import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)


from models.autoencoder import Autoencoder
from src.models.liu_dsh import LiuDSH
from src.models.supcon import ProjectionHead, SupCon
from src.models.dino import DINO
from src.models.dino import DINOv2
from src.models.fsl_models import DinoFsl
from src.models.fsl_models import DINOv2Fsl
from src.models.fsl_models import PhikonFsl
from src.models.fsl_models import ResNetFsl
from src.models.fsl_models import UNIFsl
from src.models.fsl_models import Virchow2Fsl
from src.models.fsl_models import ViTFsl
from src.models.phikon import Phikon
from src.models.resnet import ResNet, get_resnet_backbone
from src.models.uni import UNI
from src.models.virchow2 import Virchow2
from src.models.vit import ViT
from src.utils.checkpoint_utils import load_checkpoint


def get_model(model_config):
    model_code = model_config.get('model_code').lower()

    if model_code == 'resnet':
        model = ResNet(model_config)

    elif model_code == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_code == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])

    elif model_code == 'vit':
        model = ViT(model_name=model_config['model_name'])

    elif model_code == 'uni':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_code == 'UNI2-h':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_code == 'virchow2':   # Pathology Foundation Model
        model = Virchow2(model_name=model_config['model_name'])

    elif model_code == 'phikon':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])

    elif model_code == 'phikon-v2':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])
        
    ################### Benchmark Models ######################################
        
    elif model_code == 'liu_dsh':   # Deep Supervised Hashing
        model = LiuDSH(model_config)
    
    elif model_code == 'supcon':   # Supervised Contrastive Learning
        model = SupCon(model_config)

    elif model_code == "triplet":
        backbone = get_resnet_backbone(model_config)
        model = ProjectionHead(
            base_model=backbone,
            hidden_dim=model_config.get("hidden_dim", 512),
            out_dim=model_config.get("out_dim", 128),
        )

    elif model_code == "autoencoder":
        backbone = get_resnet_backbone(model_config)
        model = Autoencoder(
            backbone,
            encoder_dim=model_config.get("encoder_dim", 512),
            decoder_channels=model_config.get("decoder_channels", 512),
            decoder_h=model_config.get("decoder_h", 8),
            decoder_w=model_config.get("decoder_w", 8),
        )

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
