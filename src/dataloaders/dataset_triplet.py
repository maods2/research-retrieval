from dataloaders.dataset import StandardImageDataset
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import os
import random
import torch


class TripletDataset(StandardImageDataset):
    def __init__(
        self, root_dir, transform=None, class_mapping=None, config=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else A.Compose([A.ToFloat()])
        super().__init__(
            root_dir=root_dir,
            train_transform=transform,
            class_mapping=class_mapping
        )

        self.validation_dataset = None
        self.num_classes = len(self.class_mapping)

    def __len__(self):
        return len(self.image_paths)

    def _open_image(self, path):
        """
        Open an image file and convert it to RGB format.
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Failed to load image at {path}')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _validation__getitem__(self, idx):
        """
        Fetch a single image and its label for validation purposes.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open and transform the image
        image = self._open_image(image_path)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.validation_dataset is not None:
            return self._validation__getitem__(idx)

        anchor_path = self.image_paths[idx]
        anchor_class = self.labels[idx]

        # Choose a positive from the same class (different from the anchor)
        positive_path = random.choice(
            [p for p in self.image_dict[anchor_class] if p != anchor_path]
        )

        # Choose a negative from a different class
        negative_class = random.choice(
            [cls for cls in self.image_dict.keys() if cls != anchor_class]
        )
        negative_path = random.choice(self.image_dict[negative_class])

        # Load images using OpenCV
        anchor = cv2.cvtColor(cv2.imread(str(anchor_path)), cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(
            cv2.imread(str(positive_path)), cv2.COLOR_BGR2RGB
        )
        negative = cv2.cvtColor(
            cv2.imread(str(negative_path)), cv2.COLOR_BGR2RGB
        )

        # Apply Albumentations transformations
        if self.transform:
            anchor = self.transform(image=anchor)['image']
            positive = self.transform(image=positive)['image']
            negative = self.transform(image=negative)['image']

        return anchor, positive, negative

    def set_transform(self, transform):
        self.transform = transform
