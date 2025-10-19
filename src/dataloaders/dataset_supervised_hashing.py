import random
from torch.utils.data import Dataset
from typing import Optional, Callable, Any
import torch


class SupervisedHashingDataset(Dataset):
    """
    Dataset for supervised hashing that creates pairs of images with similarity labels.
    
    For each image, it creates a pair with another randomly selected image and
    labels whether they belong to the same class (similar) or different classes (dissimilar).
    """
    
    def __init__(
        self, 
        base_dataset: Dataset, 
        transform: Optional[Callable] = None,
        train: bool = True
    ):
        """
        Initialize the supervised hashing dataset.
        
        Args:
            base_dataset: Base dataset that provides (image, label) pairs
            transform: Optional transform to apply to images
            train: Whether this is training data (affects random sampling)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.train = train
        self.size = len(self.base_dataset)
        
        # Create class indices for efficient sampling
        self.class_indices = {}
        for idx, (_, label) in enumerate(self.base_dataset):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
    
    def __len__(self):
        return self.size
    
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
        # Get first image and label
        x_img, x_target = self.base_dataset[item]
        
        # Choose second image
        if self.train:
            # During training, randomly choose another image
            pair_idx = item
            while pair_idx == item:
                pair_idx = random.randint(0, self.size - 1)
            y_img, y_target = self.base_dataset[pair_idx]
        else:
            # During evaluation, could use different strategy
            # For now, use same random strategy
            pair_idx = item
            while pair_idx == item:
                pair_idx = random.randint(0, self.size - 1)
            y_img, y_target = self.base_dataset[pair_idx]
        
        # Apply transforms if provided
        if self.transform:
            x_img = self.transform(x_img)
            y_img = self.transform(y_img)
        
        # Create similarity label: 0 if same class, 1 if different class
        target_equals = 0 if x_target == y_target else 1
        
        return x_img, x_target, y_img, y_target, target_equals
