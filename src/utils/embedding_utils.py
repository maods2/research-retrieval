from factories.transform_factory import get_transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm
from typing import Any, Dict, Tuple, Callable, Optional

import numpy as np
import os
import time
import torch

def invert_dict(d):
    """Convert a dictionary to an inverted dictionary where keys become values and values become keys."""
    if isinstance(d, np.ndarray):
        d = d.item()  # Convert numpy array to dictionary
    return {v: k for k, v in d.items()}


def create_embeddings(
    model,
    data_loader,
    normalize_embeddings,
    device,
    logger,
    desc='Extracting features',
):
    """
    Generate embeddings and labels from a given data loader.

    Args:
        model: The model used for generating embeddings.
        data_loader: The data loader containing the data.
        device: The device to perform computations (e.g., 'cuda' or 'cpu').
        logger: Logger object for logging information.
        desc: Description for the progress bar.

    Returns:
        tuple: A tuple containing embeddings (np.ndarray) and labels (np.ndarray).
    """
    embeddings = []
    labels = []
    model.eval()
    model.to(device)
    data_loader = tqdm(data_loader, desc=desc)

    for img, label in data_loader:
        with torch.no_grad():
            embedding = model(img.to(device))
            if normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, dim=1)

        if isinstance(embedding, tuple):
            # For multi-branch models: concatenate all branch outputs along feature dimension
            embedding = torch.cat(embedding, dim=0)
        
        embeddings.append(embedding.cpu().numpy())

        # Handle both one-hot and standard labels
        if len(label.shape) > 1:  # one-hot encoded
            label = label.argmax(dim=1)
        labels.append(label.cpu().numpy())
        # labels.append(label.argmax(dim=1).cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    logger.info(
        f'Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}'
    )
    return embeddings, labels


def get_dataset_attribute(dataset, attribute_name: str):
    """
    Helper function to get attributes from either regular or subset dataset.

    Args:
        dataset: Dataset object (can be either regular dataset or subset)
        attribute_name: Name of the attribute to retrieve (e.g., 'labels', 'image_paths', 'class_mapping')

    Returns:
        The requested attribute value
    """
    if hasattr(dataset, 'dataset') and isinstance(
        dataset, Subset
    ):  # Subset dataset
        indices = dataset.indices
        original_dataset = dataset.dataset

        if attribute_name in ['image_paths', 'labels', 'labels_str']:
            # Handle list-type attributes that need to be subset
            original_attr = getattr(original_dataset, attribute_name)
            return [original_attr[i] for i in indices]
        else:
            # Handle other attributes (like class_mapping) that should be returned as-is
            return getattr(original_dataset, attribute_name)
    else:
        # Regular dataset - return attribute directly
        return getattr(dataset, attribute_name)


def compute_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute class prototypes as the mean embedding of all samples in each class.

    Args:
        embeddings: Array of shape (N, D) containing all embeddings
        labels: Array of shape (N,) containing class labels
        n_classes: Number of classes. If None, inferred from labels.max() + 1

    Returns:
        Prototypes array of shape (n_classes, D)
    """
    if n_classes is None:
        n_classes = int(labels.max()) + 1

    prototypes = np.zeros((n_classes, embeddings.shape[1]), dtype=embeddings.dtype)
    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        if class_mask.sum() > 0:
            prototypes[class_idx] = embeddings[class_mask].mean(axis=0)

    return prototypes


def concatenate_prototype_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: Optional[np.ndarray] = None,
    distance_metric: str = 'cosine',
    normalize_distances: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate embeddings with normalized distances to all class prototypes.

    Args:
        embeddings: Array of shape (N, D) containing embeddings
        labels: Array of shape (N,) containing class labels
        prototypes: Array of shape (n_classes, D) containing class prototypes.
                   If None, computed from embeddings and labels.
        distance_metric: Metric to use ('cosine', 'euclidean', 'l2')
        normalize_distances: If True, normalize similarities to [0, 1] range

    Returns:
        Tuple of:
            - Augmented embeddings of shape (N, D + n_classes)
            - Prototypes used (in case they were computed)
    """
    if prototypes is None:
        prototypes = compute_prototypes(embeddings, labels)

    n_samples, embedding_dim = embeddings.shape
    n_classes = prototypes.shape[0]

    # Compute similarities (not distances)
    if distance_metric == 'cosine':
        #from sklearn.metrics.pairwise import cosine_similarity
        #cosine_sims = cosine_similarity(embeddings, prototypes)
        similarities = embeddings @ prototypes.T
        
    elif distance_metric == 'euclidean' or distance_metric == 'l2':
        distances = np.zeros((n_samples, n_classes), dtype=embeddings.dtype)
        for class_idx in range(n_classes):
            distances[:, class_idx] = np.linalg.norm(
                embeddings - prototypes[class_idx],
                ord=2,
                axis=1
            )
        
        # Use median distance as scale to normalize
        #median_dist = np.median(distances)
        #scale = max(median_dist, 1e-6)  # Avoid division by zero
        #similarities = np.exp(-distances / scale)
        similarities = -distances
        
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    # Normalize per sample across classes
    similarities = torch.nn.functional.log_softmax(torch.from_numpy(similarities), dim=1).numpy()

    augmented_embeddings = np.concatenate([embeddings, similarities], axis=1)

    return augmented_embeddings, prototypes


def create_embeddings_dict(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    logger: Any,
    config: Dict[str, Any],
) -> Dict[str, Tuple]:
    """
    Create a dictionary containing embeddings and labels for both training and test data.

    Args:
        model: PyTorch model used for generating embeddings.
        train_loader: DataLoader for the training data.
        test_loader: DataLoader for the evaluation data.
        device: Device to perform computations ('cuda' or 'cpu').
        logger: Logger object for logging information.

    Returns:
        Dict[str, Tuple]: A dictionary with keys 'db_embeddings', 'db_labels',
                          'query_embeddings', and 'query_labels'.
    """
    if hasattr(train_loader.dataset, 'validation_dataset'):
        test_loader.dataset.validation_dataset = True
        train_loader.dataset.validation_dataset = True

    if hasattr(train_loader.dataset, 'k_shot') and hasattr(
        test_loader.dataset, 'k_shot'
    ):
        train_loader.dataset.k_shot = 1
        test_loader.dataset.k_shot = 1

    # set transforms to test transforms for eval
    train_loader.dataset.set_transform(
        get_transforms(config['transform'].get('test', None))
    )

    logger.info('Creating embeddings database from training data...')
    normalize_embeddings = config['evaluation'].get(
        'normalize_embeddings', False
    )
    db_embeddings, db_labels = create_embeddings(
        model,
        train_loader,
        normalize_embeddings,
        device,
        logger,
        desc='Creating database',
    )

    logger.info('Generating query embeddings from test data...')
    query_embeddings, query_labels = create_embeddings(
        model,
        test_loader,
        normalize_embeddings,
        device,
        logger,
        desc='Generating queries',
    )

    # Concatenate embeddings with prototype distances
    prototypes = None
    if config['evaluation'].get('augment_with_prototype_distances', False):
        logger.info('Computing prototypes and concatenating distances...')
        distance_metric = config['evaluation'].get(
            'prototype_distance_metric', 'euclidean'
        )
        
        # Compute prototypes from training data
        prototypes = compute_prototypes(db_embeddings, db_labels)
        
        # Concatenate DB embeddings with distances to prototypes
        db_embeddings, _ = concatenate_prototype_distances(
            db_embeddings,
            db_labels,
            prototypes=prototypes,
            distance_metric=distance_metric,
        )
        
        # Concatenate query embeddings with distances to prototypes
        query_embeddings, _ = concatenate_prototype_distances(
            query_embeddings,
            query_labels,
            prototypes=prototypes,
            distance_metric=distance_metric,
        )
        
        logger.info(
            f'Augmented DB embeddings shape: {db_embeddings.shape}'
        )
        logger.info(
            f'Augmented query embeddings shape: {query_embeddings.shape}'
        )

    # Use the generalist function to get attributes
    embeddings = {
        'db_embeddings': db_embeddings,
        'db_labels': db_labels,
        'db_path': get_dataset_attribute(train_loader.dataset, 'image_paths'),
        'query_embeddings': query_embeddings,
        'query_labels': query_labels,
        'query_classes': get_dataset_attribute(
            test_loader.dataset, 'labels_str'
        ),
        'query_paths': get_dataset_attribute(
            test_loader.dataset, 'image_paths'
        ),
        'class_mapping': invert_dict(
            get_dataset_attribute(train_loader.dataset, 'class_mapping')
        ),
    }
    
    # Store prototypes if computed
    if prototypes is not None:
        embeddings['prototypes'] = prototypes

    if config['evaluation']['save_embeddings']:
        # Ensure the directory exists
        os.makedirs(config['workspace_dir'], exist_ok=True)

        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(
            config['workspace_dir'],
            f'embeddings_{timestamp}.npz',
        )
        np.savez(path, **embeddings)
        logger.info(f'Embeddings saved to {path}')
        return embeddings, path

    return embeddings, None


def load_or_create_embeddings(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    logger: Any,
    device: str = None,
) -> Tuple[Dict, str]:
    """
    Load existing embeddings or create new ones based on configuration.

    Args:
        model: PyTorch model to use for creating embeddings
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        config: Configuration dictionary containing evaluation settings
        logger: Logger instance for logging information
        device: Device to use for computation (default: None, will use cuda if available)

    Returns:
        Tuple containing:
            - Dictionary of embeddings
            - String path where embeddings were saved (if created)
    """
    if device is None:
        device = config.get(
            'device', 'cuda' if torch.cuda.is_available() else 'cpu'
        )
    logger.info(f'Using device: {device}')

    if config['evaluation'].get('load_embeddings', False):
        return load_embeddings(config['evaluation']['embeedings_path'], logger)

    logger.info('Creating new embeddings...')
    embeddings, file_path = create_embeddings_dict(
        model, train_loader, test_loader, device, logger, config
    )
    config['evaluation']['embeddings_path'] = file_path
    return embeddings

def load_embeddings(embeddings_path: str, logger: Callable = print) -> tuple[np.lib.npyio.NpzFile, str]:
    try:
        logger = logger.info
    except AttributeError:
        pass

    logger(
        f"Loading embeddings from {embeddings_path}"
    )

    embeddings = np.load(
        embeddings_path, allow_pickle=True
    )
    return embeddings, embeddings_path