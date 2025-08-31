import os
import sys
from typing import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import cv2
import torch
import numpy as np


# I feel like I overcomplicated a bit, this may not the best solution.
# Perhaps keeping train/test/val sets separate may be the best approach,
# but this approach with views may allow for easier partitioning of the dataset.
class DatasetView(torch.utils.data.Dataset):
    def __init__(
        self, 
        parent_dataset,
        idxs: list[int],
        transform: Optional[Callable] = None
    ):
        self.parent = parent_dataset
        self.idxs = idxs
        self.transform = transform

    def set_transform(self, new_transform: Callable):
        self.transform = new_transform
        
    def __len__(self) -> int:
        return len(self.idxs)
            
    def __getitem__(self, idx):
        out = self.parent[self.idxs[idx]]
        if self.transform is not None:
            out[0] = self.transform(image=out[0])['image']
        return out

class StandardImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        return_one_hot: bool = False,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
        config: Optional[dict[str, Any]] = None, 
        test_transform: Optional[Callable] = None,
        train_transform: Optional[Callable] = None,
        class_mapping: Optional[dict[str, int]] = None,
        shuffle_generator: Optional[np.random.Generator] = None,
    ):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Root directory containing class subdirectories.
            return_one_hot (bool, optional): Returns labels as a one-hot encoded vector. Default: False.
            test_split (float, optional): Percentage of samples to be used for testing. If none,
                                          won't split into train/test. Default: None.
            val_split  (float, optional): Percentage of training samples to be used for validation. If none,
                                          won't split into train/val. Default: None.
            train_transform (callable, optional): Image transformations to be applied to training samples. 
                                          Default: None.
            test_transform (callable, optional): Image transformations to be applied to test/validation samples.
                                          Default: None.
            class_mapping (dict, optional): Custom mapping for classes. If none, it will be automatically
                                            created based on folder names. Default: None.
            shuffle_generator (np.random.Generator, optional): Generator to be used for shuffling the dataset
                                            samples. If none, won't shuffle. Default: None.
        """
        self.root_dir = Path(root_dir)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.image_paths = []       # List of full paths to images
        self.labels = []            # List of integer labels
        self.one_hot_labels = []    # List of one-hot encoded labels
        self.labels_str = []        # List of text labels
        self.return_one_hot = return_one_hot
        self.shuffle_generator = shuffle_generator

        self.test_split = test_split
        self.val_split  = val_split
        if config is not None:
            if 'data' in config.keys():
                if self.test_split is None and 'train_test_split' in config['data'].keys():
                    self.test_split = config['data']['train_test_split']
                if self.val_split is None and 'train_val_split' in config['data'].keys():
                    self.val_split = config['data']['train_val_split']
            else:
                print("[WARNING] 'data' not found in config file, skipping dataset splitting.")

        self.subsets: dict[str, DatasetView] = {} # maps subset names (train, test, val) to DatasetView obj.

        #TODO: Maybe make this a bit more flexible, e.g. load train/test from folders and dynamically split train/val. 
        #TODO: Maybe support dynamic creation of subsets from folders?
        #### Load train/test/val folders individually 
        if 'train' in os.listdir(self.root_dir):
            print("Detected `train` folder in dir. Loading subsets (train/test/val) based on folder structure.")
            self._load_data_from_dir(self.root_dir / 'train', class_mapping=class_mapping)
            self.subsets['train'] = self.create_view(
                idxs=np.arange(len(self.image_paths)),
                transform=self.train_transform
            )

            if (self.root_dir / 'val').exists():
                self._load_data_from_dir(self.root_dir / 'val', class_mapping=class_mapping)
                self.subsets['val'] = self.create_view(
                    idxs=np.arange(start=len(self.subsets['train']), stop=len(self.image_paths)),
                    transform=self.test_transform
                )

            self._load_data_from_dir(self.root_dir / 'test', class_mapping=class_mapping)
            self.subsets['test'] = self.create_view(
                idxs=np.arange(start=len(self.subsets['train'])+(len(self.subsets['val']) if 'val' in self.subsets.keys() else 0), stop=len(self.image_paths)),
                transform=self.test_transform
            )

        else: ##### Dynamic train/test/val splitting
            print("`train` folder not found in specified directory. Loading images and splitting " \
                  "into train/test/val subsets based on provided values.")
            self._load_data_from_dir(self.root_dir, class_mapping=class_mapping)        
            assert len(self.image_paths) == len(self.labels)

            self.split_into_subsets(
                test_split=self.test_split,
                val_split=self.val_split,
                shuffle_generator=self.shuffle_generator,
            )

    def split_into_subsets(
        self, 
        test_split: Optional[float] = None, 
        val_split: Optional[float] = None, 
        shuffle_generator: Optional[np.random.Generator] = None
    ):
            # list of indices. Made this way to avoid messing with positioning
            # in multiple lists at once. This way we can easily retrieve images,
            # labels, label_str and one_hot from their indices.
            idxs = np.arange(len(self.image_paths))

            if shuffle_generator is not None:
                shuffle_generator.shuffle(idxs)
            
            # train/test split
            if test_split is not None:
                train_split = int(np.floor((1-test_split)*idxs.size))
                self.subsets['test']  = self.create_view(
                    idxs=idxs[train_split:],
                    transform=self.test_transform
                )
                self.subsets['train'] = self.create_view(
                    idxs=idxs[:train_split],
                    transform=self.train_transform
                )
                print(f"Cut dataset ({len(idxs)} samples) into {train_split*100/len(idxs):.1f}% training images ({len(self.subsets['train'])} samples)",
                      f"and {len(self.subsets['test'])*100/len(idxs):.1f}% test images ({len(self.subsets['test'])} samples).")

            # train/val split
            if val_split is not None:
                train_split = int(np.floor((1-val_split)*len(self.subsets['train'])))
                self.subsets['val']   = self.create_view(
                    idxs=self.subsets['train'].idxs[train_split:],
                    transform=self.test_transform
                )
                self.subsets['train'] = self.create_view(
                    idxs=self.subsets['train'].idxs[:train_split],
                    transform=self.train_transform
                )
                print(f"Cut training set ({len(idxs)} samples) into {len(self.subsets['train'])*100/(len(self.subsets['train']+self.subsets['val'])):.1f}% training images ({len(self.subsets['train'])} samples)",
                      f"and {len(self.subsets['val'])*100/(len(self.subsets['train']+self.subsets['val'])):.1f}% validation images ({len(self.subsets['val'])} samples).")

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

    def __len__(self) -> int:
        # Return the total number of samples in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returns an image and its corresponding label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (transformed image, one-hot encoded label)
        """
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

        return image, label

    def create_view(self, idxs: list[int], transform: Optional[Callable] = None) -> DatasetView:
        return DatasetView(
            parent_dataset=self,
            idxs=idxs,
            transform=transform
        )

    def train(self) -> DatasetView:
        return self.subsets['train']

    def test(self) -> DatasetView:
        return self.subsets['test']

    def validation(self) -> DatasetView:
        return self.subsets['val']

    def set_train_transform(self, transform: Callable):
        """
        override the base class method to set a new train transform.
        """
        self.subsets['train'].set_transform(transform)

    def set_test_transform(self, transform: Callable):
        """
        override the base class method to set a new test transform.
        """
        self.subsets['test'].set_transform(transform)
        self.subsets['val'].set_transform(transform)