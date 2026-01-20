import os
import sys
from typing import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import torch


from dataloaders.dataset import StandardImageDataset

class ContrastiveDataset(StandardImageDataset):
    """
    Returns two randomly augmented views per sample, using an optional base transform
    """

    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[Callable] = None, 
        class_mapping: Optional[dict[str, int]] = None, 
        config: Optional[dict[str, Any]] = None
    ):
        super().__init__(
            root_dir=root_dir, 
            transform=None, 
            class_mapping=class_mapping, 
            config=config
        )
        self._transform = transform
        self.validation_dataset = None

    def _open_image(self, path):
        """
        Open an image file and convert it to RGB format.
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Failed to load image at {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def __getitem__(self, idx):
        if self.validation_dataset is not None:
            return self._validation__getitem__(idx)

        image, label = super().__getitem__(idx)

        if self._transform:
            img1 = self._transform(image=image)['image']
            img2 = self._transform(image=image)['image']
        else:
            img1 = img2 = image

        return torch.stack([img1, img2]), label

    def set_transform(self, transform):
        """
        ovwerride the base class method to set a new transform.
        """
        self._transform = transform
