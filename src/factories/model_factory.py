import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)


from src.models.autoencoder import Autoencoder
from src.models.clip import CLIP
from src.models.dino import DINO
from src.models.dino import DINOv2
from src.models.fsl_models import WrappedFsl
from src.models.phikon import Phikon
from src.models.resnet import ResNet18
from src.models.resnet import ResNet34
from src.models.resnet import ResNet50
from src.models.supcon_models import ProjectionHead
from src.models.triplet_resnet import ResNet
from src.models.triplet_resnet import TripletResNet
from src.models.uni import UNI
from src.models.virchow2 import Virchow2
from src.models.vit import TripletViT
from src.models.vit import ViT
from src.utils.checkpoint_utils import load_checkpoint


MODEL_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


from src.models.mynet import MyNet  # Import after registry is defined


def get_model(model_config):
    model_name = model_config.get('model_name', model_config.get('name'))
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](**model_config)

    if model_name == 'resnet':
        model = ResNet(model_name=model_config['model_name'])

    elif model_name == 'dino':
        model = DINO(model_name=model_config['model_name'])

    elif model_name == 'dinov2':
        model = DINOv2(model_name=model_config['model_name'])

    elif model_name == 'clip':
        model = CLIP(model_name=model_config['model_name'])

    elif model_name == 'vit':
        model = ViT(model_name=model_config['model_name'])

    elif model_name == 'uni':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'UNI2-h':   # Pathology Foundation Model
        model = UNI(model_name=model_config['model_name'])

    elif model_name == 'virchow2':   # Pathology Foundation Model
        model = Virchow2(model_name=model_config['model_name'])

    elif model_name == 'phikon':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])

    elif model_name == 'phikon-v2':   # Pathology Foundation Model
        model = Phikon(model_name=model_config['model_name'])

    ################### Few-Shot Learning Models ######################################

    elif model_name == 'dino_fsl':
        backbone = DINO(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'dinov2_fsl':
        backbone = DINOv2(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'clip_fsl':
        backbone = CLIP(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'vit_fsl':
        backbone = ViT(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'uni_fsl':   # Pathology Foundation Model
        backbone = UNI(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'UNI2-h_fsl':   # Pathology Foundation Model
        backbone = UNI(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'virchow2_fsl':   # Pathology Foundation Model
        backbone = Virchow2(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'resnet_fsl':   # Pathology Foundation Model
        backbone = ResNet(model_name=model_config['arch_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'phikon_fsl':   # Pathology Foundation Model
        backbone = Phikon(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    elif model_name == 'phikon-v2_fsl':   # Pathology Foundation Model
        backbone = Phikon(model_name=model_config['model_name'])
        model = WrappedFsl(
            backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            embedding_dim=model_config.get('embedding_dim', 128),
        )

    ################### Triplet Models #################################################

    elif model_name == 'triplet_resnet':
        if model_config.get('model_name') == 'resnet18':
            backbone = ResNet18(model_config)
        elif model_config.get('model_name') == 'resnet20':
            backbone = ResNet34(model_config)
        elif model_config.get('model_name') == 'resnet50':
            backbone = ResNet50(model_config)
        else:
            raise ValueError(
                f'Model {model_config["model_name"]} is not supported'
            )
        model = ProjectionHead(
            base_model=backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            out_dim=model_config.get('out_dim', 128),
        )

    elif model_name == 'triplet_vit':
        model = TripletViT(embedding_size=model_config['embedding_size'])

    #################### Other Models ######################################################
    elif model_name == 'resnet_supcon':
        if model_config.get('model_name') == 'resnet18':
            backbone = ResNet18(model_config)
        elif model_config.get('model_name') == 'resnet20':
            backbone = ResNet34(model_config)
        elif model_config.get('model_name') == 'resnet50':
            backbone = ResNet50(model_config)
        else:
            raise ValueError(
                f'Model {model_config["model_name"]} is not supported'
            )

        model = ProjectionHead(
            base_model=backbone,
            hidden_dim=model_config.get('hidden_dim', 512),
            out_dim=model_config.get('out_dim', 128),
        )

    elif model_name == 'autoencoder_resnet':
        if model_config.get('model_name') == 'resnet18':
            backbone = ResNet18(model_config)
        elif model_config.get('model_name') == 'resnet20':
            backbone = ResNet34(model_config)
        elif model_config.get('model_name') == 'resnet50':
            backbone = ResNet50(model_config)
        else:
            raise ValueError(
                f'Model {model_config["model_name"]} is not supported'
            )

        model = Autoencoder(
            backbone,
            encoder_dim=model_config.get('encoder_dim', 512),
            decoder_channels=model_config.get('decoder_channels', 512),
            decoder_h=model_config.get('decoder_h', 8),
            decoder_w=model_config.get('decoder_w', 8),
        )

    elif model_name == 'mynet':
        model = MyNet(**model_config)

    else:
        raise ValueError(f'Model {model_name} is not supported')

    if model_config['load_checkpoint']:
        load_checkpoint(model_config['checkpoint_path'], model)

    return model
