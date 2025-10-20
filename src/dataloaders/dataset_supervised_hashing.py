import random
import os
import sys
from typing import Optional, Callable, Any
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataloaders.dataset import StandardImageDataset


class SupervisedHashingDataset(StandardImageDataset):
    """
    Dataset for supervised hashing that creates pairs of images with similarity labels.
    
    Inherits from StandardImageDataset to follow the framework's configuration pattern.
    For each image, it creates a pair with another randomly selected image and
    labels whether they belong to the same class (similar) or different classes (dissimilar).
    """
    
    def __init__(
        self, 
        root_dir,
        transform=None,
        class_mapping=None,
        config=None,
        return_one_hot=False,
        train: bool = True
    ):
        """
        Initialize the supervised hashing dataset.
        
        Args:
            root_dir (str): Root directory containing class subdirectories.
            transform (callable, optional): Transformations to be applied to the images.
            class_mapping (dict, optional): Custom mapping for classes.
            config (dict, optional): Configuration dictionary.
            return_one_hot (bool): Whether to return one-hot encoded labels.
            train (bool): Whether this is training data (affects random sampling).
        """
        # Initialize parent class
        super().__init__(
            root_dir=root_dir,
            transform=transform,
            class_mapping=class_mapping,
            config=config,
            return_one_hot=return_one_hot
        )
        
        self.train = train
        self.validation_dataset = None
        # Create class indices for efficient sampling
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        """
        Get a pair of images with similarity label.
        
        Returns:
            x_img: First image
            x_target: Label of first image
            y_img: Second image  
            y_target: Label of second image
            target_equals: 0 if same class, 1 if different class
        """
        if self.validation_dataset is not None:
            return self._validation__getitem__(item)
        
        # Get first image and label using parent class method
        x_img, x_target = super().__getitem__(item)
        
        # Choose second image
        if self.train:
            # During training, randomly choose another image
            pair_idx = item
            while pair_idx == item:
                pair_idx = random.randint(0, len(self) - 1)
            y_img, y_target = super().__getitem__(pair_idx)
        else:
            # During evaluation, could use different strategy
            # For now, use same random strategy
            pair_idx = item
            while pair_idx == item:
                pair_idx = random.randint(0, len(self) - 1)
            y_img, y_target = super().__getitem__(pair_idx)
        
        # Create similarity label: 0 if same class, 1 if different class
        target_equals = 0 if x_target == y_target else 1
        
        return x_img, x_target, y_img, y_target, target_equals
    
    def set_transform(self, transform):
        """
        Override the parent class method to set a new transform.
        """
        self.transform = transform
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset.
        
        Returns:
            dict: Dictionary mapping class labels to their counts
        """
        from collections import Counter
        return dict(Counter(self.labels))
    
    def get_samples_by_class(self, class_label, num_samples=None):
        """
        Get samples from a specific class.
        
        Args:
            class_label: The class label to sample from
            num_samples: Number of samples to return (None for all)
            
        Returns:
            list: List of (image, label) tuples
        """
        if class_label not in self.class_indices:
            raise ValueError(f"Class {class_label} not found in dataset")
        
        indices = self.class_indices[class_label]
        if num_samples is not None:
            indices = random.sample(indices, min(num_samples, len(indices)))
        
        return [super().__getitem__(idx) for idx in indices]


if __name__ == '__main__':
    """
    Example usage of SupervisedHashingDataset.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    
    # Example root directory (adjust path as needed)
    root_dir = './data/example_dataset/train'
    
    # Define transformations using Albumentations
    data_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    
    # Create the dataset
    try:
        dataset = SupervisedHashingDataset(
            root_dir=root_dir,
            transform=data_transforms,
            train=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        
        # Test the dataset
        for batch in dataloader:
            x_imgs, x_targets, y_imgs, y_targets, target_equals = batch
            print(f"Batch shapes:")
            print(f"  x_imgs: {x_imgs.shape}")
            print(f"  x_targets: {x_targets.shape}")
            print(f"  y_imgs: {y_imgs.shape}")
            print(f"  y_targets: {y_targets.shape}")
            print(f"  target_equals: {target_equals.shape}")
            print(f"  Similar pairs: {(target_equals == 0).sum().item()}")
            print(f"  Dissimilar pairs: {(target_equals == 1).sum().item()}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the root_dir path exists and contains class subdirectories.")
