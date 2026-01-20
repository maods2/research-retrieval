import torch.nn as nn

import losses as L

def get_loss(loss_config):
    loss_name = loss_config['name']

    if not loss_name:
        return None
    elif loss_name == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()  # For multilabel classification
    elif loss_name == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()  # For multiclass classification
    elif loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'adaptative_triplet':
        loss_fn = L.AdaptiveTripletLoss()
    elif loss_name == 'prototypical':
        loss_fn = L.PrototypicalLoss(loss_config)
    elif loss_name == 'ntxent':
        loss_fn = L.NTXentLoss(loss_config)
    elif loss_name == 'supervised_contrastive':
        loss_fn = L.SupConLoss(loss_config)
    elif loss_name == 'proxy_nca':
        loss_fn = L.ProxyNCALoss(loss_config)
    elif loss_name == 'multi_similarity':
        loss_fn = L.MultiSimilarityLoss(loss_config)
    elif loss_name == 'arcface':
        loss_fn = L.ArcFaceLoss(loss_config)
    elif loss_name == 'npair':
        loss_fn = L.NPairLoss(loss_config)
    elif loss_name == 'supervised_hashing':
        loss_fn = L.SupervisedHashingLoss(loss_config)
    elif loss_name in ('sca', 'supervised_contrastive_attention'):
        loss_fn = L.SupervisedContrastiveAttention(loss_config)

    else:
        raise ValueError(f'Loss function {loss_name} is not supported')

    return loss_fn
