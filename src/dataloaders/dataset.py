import os
import sys
from typing import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path
from torch.utils.data import Dataset

import cv2
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split

# I feel like I overcomplicated a bit, this may not the best solution.
# Perhaps keeping train/test/val sets separate may be the best approach,
# but this approach with views may allow for easier partitioning of the dataset.
class DatasetView(Dataset):
    def __init__(self, std_img_dataset, subset: str):
        self.parent = std_img_dataset
        self.subset = subset
        assert self.subset in self.parent.subsets.keys()
        
    def __len__(self):
        return len(self.parent.subsets[self.subset])
            
    def __getitem__(self, idx):
        translated_idx = self.parent.subsets[self.subset][idx]
        return self.parent[translated_idx]

class StandardImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        return_one_hot: bool = False,
        transform: Optional[Callable] = None,
        config: Optional[dict[str, Any]] = None, 
        class_mapping: Optional[dict[str, int]] = None,
        test_split: Optional[float] = None,
        val_split: Optional[float] = None,
        shuffle_generator: Optional[np.random.Generator] = None,
    ):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory containing class subdirectories.
            return_one_hot (bool): Returns labels as a one-hot encoded vector. Default: False.
            transform (callable, optional): Transformations to be applied to the images.
            class_mapping (dict, optional): Custom mapping for classes.
                                            If None, it will be automatically created based on folder names.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []       # List of full paths to images
        self.labels = []            # List of integer labels
        self.one_hot_labels = []    # List of one-hot encoded labels
        self.labels_str = []        # List of text labels
        self.return_one_hot = return_one_hot
        self.shuffle_generator = shuffle_generator

        self.subsets = {} # maps subset names (train, test, val) to a list of self.image_paths/self.labels indices.

        #### Load train/test/val folders if both_test_split
        #TODO: Maybe make this a bit more flexible, e.g. load train/test from folders and dynamically split train/val. 
        #TODO: Maybe support dynamic creation of subsets from folders?
        if 'train' in os.listdir(self.root_dir):
            self._load_data_from_dir(self.root_dir / 'train', class_mapping=class_mapping)
            self.subsets['train'] = np.arange(len(self.image_paths))

            self.val_idxs = []
            if (self.root_dir / 'val').exists():
                self._load_data_from_dir(self.root_dir / 'val', class_mapping=class_mapping)
                self.subsets['val'] = np.arange(start=len(self.subsets['train']), stop=len(self.image_paths))

            self._load_data_from_dir(self.root_dir / 'test', class_mapping=class_mapping)
            self.subsets['test'] = np.arange(start=len(self.subsets['train'])+len(self.val_idxs), stop=len(self.image_paths))

        else: ##### Dynamic train/test/val splitting
            self.test_split = test_split
            self.val_split  = val_split

            self._load_data_from_dir(self.root_dir, class_mapping=class_mapping)        
            assert len(self.image_paths) == len(self.labels)

            # list of indices. Made this way to avoid messing with positioning
            # in multiple lists at once. This way we can easily retrieve images,
            # labels, label_str and one_hot from their indices.
            idxs = np.arange(len(self.image_paths))

            if self.shuffle_generator is not None:
                self.shuffle_generator.shuffle(idxs)
            
            if test_split is not None:
                train_split = int(np.floor((1-test_split)*idxs.size))
                self.subsets['train'], self.subsets['test'] = idxs[:train_split], idxs[train_split:]

            if val_split is not None:
                train_split = int(np.floor((1-val_split)*self.subsets['train'].size))
                self.subsets['train'], self.subsets['val'] = self.subsets['train'][:train_split], self.subsets['train'][train_split:]


    def _load_data_from_dir(self, root_dir: Path, class_mapping: Optional[dict]):

        # Get class names from folder names
        self.classes = sorted(
            [
                folder.name
                for folder in root_dir.iterdir()
                if folder.is_dir()
            ]
        ) if class_mapping is None else list(class_mapping.keys())

        # Create automatic mapping if none is provided
        if class_mapping is None:
            class_mapping = {
                class_name: idx for idx, class_name in enumerate(self.classes)
            }

        # Validate class mapping; asserts every class is contained in a mapping.
        for class_name in self.classes:
            validation = [False for _ in range(len(self.classes))]
            for i, key_name in enumerate(class_mapping.keys()):
                if key_name in class_name:
                    validation[i] = True

            if not any(validation):
                raise ValueError(
                    f'Class {class_name} not found in class mapping'
                )
                

        self.class_mapping = class_mapping
        self.image_dict = {
            self.class_mapping[class_name]: [] for class_name in self.classes
        }

        for class_name in self.classes:
            image_extensions = [
                '*.jpg',
                '*.jpeg',
                '*.png',
                '*.tif',
                '*.tiff',
                '*.JPG',
                '*.JPEG',
                '*.PNG',
                '*.TIF',
                '*.TIFF',
                '*.webp',
            ]

            # Collect all matching image files for each extension
            images = []
            for folder in root_dir.iterdir():
                if class_name in folder.name: # any folder that contains `class_name` will be considered.
                    for ext in image_extensions:
                        images.extend(folder.rglob(ext))

            self.image_dict[self.class_mapping[class_name]].extend(images)

            # Register images and labels
            for file_path in images:

                # Create one-hot encoding for the class
                one_hot_label = np.zeros(
                    len(self.class_mapping), dtype=np.float32
                )
                one_hot_label[self.class_mapping[class_name]] = 1.0

                self.image_paths.append(file_path)
                self.labels.append(self.class_mapping[class_name])
                self.one_hot_labels.append(one_hot_label)
                self.labels_str.append(class_name)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns an image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (transformed image, one-hot encoded label)
        """
        print("RECEIVED IDX:", idx)
        img_path = str(self.image_paths[idx])

        if self.return_one_hot:
            label = torch.tensor(self.one_hot_labels[idx], dtype=torch.float32)
        else:
            label = self.labels[idx]

        # Load the image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f'Failed to load image at {img_path}')

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations, if provided
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

    def get_subset(self, subset: str) -> DatasetView:
        return DatasetView(self, subset)

    def train(self) -> DatasetView:
        return self.get_subset('train')

    def test(self) -> DatasetView:
        return self.get_subset('test')

    def validation(self) -> DatasetView:
        return self.get_subset('val')

    def set_transform(self, transform):
        """
        ovwerride the base class method to set a new transform.
        """
        self.transform = transform
