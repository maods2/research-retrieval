import torch.optim as optim

try:
    from pathology_foundation_models.models import FoundationModel
    PFM_AVAILABLE = True
except ImportError:
    FoundationModel = None
    PFM_AVAILABLE = False


def get_optimizer(optimizer_config, model):
    optimizer_name = optimizer_config['name']
    lr = float(optimizer_config['lr'])
    weight_decay = float(optimizer_config.get('weight_decay', 0.0))

    model_params = model.model.parameters() if (PFM_AVAILABLE and isinstance(model, FoundationModel)) else model.parameters()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model_params)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(params=model_params)
    elif optimizer_name == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        optimizer = optim.SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f'Optimizer {optimizer_name} is not supported')

    return optimizer
