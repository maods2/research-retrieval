from typing import Optional, Callable

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import pathology_foundation_models as pfm

from dataloaders.dataset import StandardImageDataset
from dataloaders.dataset_contrastive import ContrastiveDataset
from dataloaders.dataset_fewshot import FixedFewshotFolderDataset, VariableFewShotFolderDataset
from dataloaders.dataset_triplet import TripletDataset
from dataloaders.dataset_embedding import EmbeddingDataset
from dataloaders.dataset_embedding_precomputed import PrecomputedEmbeddingDataset

from factories.model_factory import get_model
from utils.auth_utils import get_hf_token

def get_dataset(config, transform_train: Optional[Callable] = None, transform_test: Optional[Callable] = None):
    data_config = config['data']  # Extract data config from the main config

    # Get transformations based on config
    dataset_name = data_config.get(
        'dataset_type', 'StandardImageDataset'
    )  # Default to TerumoImageDataset

    # Select dataset class dynamically based on config
    match (dataset_name):
        case 'StandardImageDataset':
            dataset_class = StandardImageDataset
        case 'TripletDataset':
            dataset_class = TripletDataset
        case 'FewShotFolderDataset':
            if config['data'].get('fixed_support_set', False):
                dataset_class = FixedFewshotFolderDataset 
            else:
                dataset_class = VariableFewShotFolderDataset
        case 'ContrastiveDataset':
            dataset_class = ContrastiveDataset
        case 'EmbeddingDataset':
            dataset_class = EmbeddingDataset
        case 'PrecomputedEmbeddingDataset':
            dataset_class = PrecomputedEmbeddingDataset
        case _:
            raise ValueError(f'Dataset {dataset_name} is not supported.')

    assert dataset_class is not None, "Unreachable"
    if dataset_class is PrecomputedEmbeddingDataset:
        # special case, no image directories, load from npz files
        train_dataset = dataset_class.from_npz(
            npz_path=data_config['train_npz_path'],
            class_mapping=data_config['class_mapping'],
        )

        test_dataset = dataset_class.from_npz(
            npz_path=data_config['test_npz_path'],
            class_mapping=data_config['class_mapping'],
        )
        return train_dataset, test_dataset

    # if dataset_class != EmbeddingDataset:
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
    # else: # HACK
    #     embedding_dataset = EmbeddingDataset( 
    #         root_dir=data_config['data_path'],
    #         transform=transform_train,
    #         class_mapping=data_config['class_mapping'],
    #         config=config,
    #         test_split=data_config['train_test_split']
    #     )
    #     train_dataset = torch.utils.data.Subset(
    #         embedding_dataset,
    #         embedding_dataset.train_idxs
    #     )
    #     test_dataset = torch.utils.data.Subset(
    #         embedding_dataset,
    #         embedding_dataset.test_idxs
    #     )
    #     return train_dataset, test_dataset
        


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
