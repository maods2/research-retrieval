from typing import Optional, Callable

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pathology_foundation_models as pfm

from dataloaders.dataset import StandardImageDataset
from dataloaders.dataset_contrastive import ContrastiveDataset
from dataloaders.dataset_fewshot import FewShotFolderDataset
from dataloaders.dataset_triplet import TripletDataset
from src.utils.auth_utils import get_hf_token

def get_embedding_dataset(config: dict, transform_train, transform_test):
    new_cfg = config.copy()
    new_cfg['data']['dataset_type'] = 'StandardImageDataset'
    train_img_set, test_img_set = get_dataset(new_cfg, transform_train, transform_test)
    del new_cfg

    train_embed_dataset = pfm.dataset.EmbeddingCache.init_from_image_dataset(
        image_dataset = train_img_set,
        model = pfm.models.load_foundation_model(
            model_type = config['model']['model_name'],
            device = 'cuda' if torch.cuda.is_available() else 'cpu', 
            token = get_hf_token()
        ),
        batch_size = config['data']['batch_size_train'],
        num_workers = config['data']['num_workers']
    )

    test_embed_dataset = pfm.dataset.EmbeddingCache.init_from_image_dataset(
        image_dataset = test_img_set,
        model = pfm.models.load_foundation_model(
            model_type = config['model']['model_name'],
            device = 'cuda' if torch.cuda.is_available() else 'cpu', 
            token = get_hf_token()
        ),
        batch_size = config['data']['batch_size_test'],
        num_workers = config['data']['num_workers']
    )

    return train_embed_dataset, test_embed_dataset

def get_dataset(config, transform_train: Optional[Callable] = None, transform_test: Optional[Callable] = None):
    data_config = config['data']  # Extract data config from the main config

    # Get transformations based on config
    dataset_name = data_config.get(
        'dataset_type', 'StandardImageDataset'
    )  # Default to TerumoImageDataset

    # Select dataset class dynamically based on config
    if dataset_name == 'StandardImageDataset':
        dataset_class = StandardImageDataset
    elif dataset_name == 'TripletDataset':
        dataset_class = TripletDataset
    elif dataset_name == 'FewShotFolderDataset':
        dataset_class = FewShotFolderDataset
    elif dataset_name == 'ContrastiveDataset':
        dataset_class = ContrastiveDataset
    elif dataset_name == 'EmbeddingDataset':
        return get_embedding_dataset(config, transform_train, transform_test)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')

    # Create dataset instances for training and evaluation
    train_dataset = dataset_class(
        root_dir=data_config['train_dir'],  # Directory for training data
        transform=transform_train,  # Transformations to apply
        class_mapping=data_config['class_mapping'],  # Custom class mappings
        config=config,  # Additional config for dataset
    )

    test_dataset = dataset_class(
        root_dir=data_config['test_dir'],  # Directory for evaluation data
        transform=transform_test,  # Transformations to apply
        class_mapping=data_config['class_mapping'],  # Custom class mappings
        config=config,  # Additional config for dataset
    )

    return train_dataset, test_dataset

def get_dataloader(config, transform_train: Optional[Callable] = None, transform_test: Optional[Callable] = None):
    """
    Factory function to get data loaders for train and test sets.

    Args:
        config (dict): Configuration dict containing information on data directories,
                             transformations, and other parameters.
        transform (A.Compose): Composed Albumentations transformation pipeline.

    Returns:
        train_loader (DataLoader): PyTorch DataLoader for the training set
        test_loader (DataLoader): PyTorch DataLoader for the test set
    """
    data_config = config['data']  # Extract data config from the main config

    train_dataset, test_dataset = get_dataset(
        config=config,
        transform_train=transform_train,
        transform_test=transform_test
    )

    # Create DataLoader instances for both datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size_train'],  # Define the batch size
        shuffle=data_config.get('shuffle_train', True),  # Shuffle for training
        num_workers=data_config[
            'num_workers'
        ],  # Number of workers for data loading
        pin_memory=True,  # Pin memory for faster data transfer
    )

    batch_size_eval = data_config.get('batch_size_eval', None)
    if batch_size_eval is None:
        batch_size_eval = data_config['batch_size_train']

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,  # Define the batch size
        shuffle=data_config.get(
            'shuffle_test', False
        ),  # No need to shuffle test set
        num_workers=data_config[
            'num_workers'
        ],  # Number of workers for data loading
        pin_memory=True,  # Pin memory for faster data transfer
    )

    return train_loader, test_loader
