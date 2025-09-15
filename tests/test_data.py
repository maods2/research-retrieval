import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataloaders import (
    StandardImageDataset,
    FewShotFolderDataset,
    TripletDataset,
    ContrastiveDataset,
)

TRANSFORMS = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)
CUSTOM_MAPPING = {
    'Crescent': 0,
    'Hypercelularidade': 1,
    'Membranous': 2,
    'Normal': 3,
    'Podocitopatia': 4,
    'Sclerosis': 5,
}

def test_standardimagedataset():
    root_dir = '/datasets/terumo-data-jpeg'
    # Create the dataset
    dataset = StandardImageDataset(
        root_dir=root_dir,
        transform=TRANSFORMS,
        class_mapping=CUSTOM_MAPPING,
        test_split=0.2,
        val_split=0.1,
        shuffle_generator=np.random.default_rng(seed=1)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    # Test the dataset
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break

def test_fewshot_dataset():
    root_dir = '/datasets/glomerulus-split/'
    config = {'model': {'n_way': 6, 'k_shot': 5, 'q_queries': 6}}

    # Create the dataset
    dataset = FewShotFolderDataset(
        root_dir=root_dir,
        transform=TRANSFORMS,
        class_mapping=CUSTOM_MAPPING,
        config=config,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )
    test_loader = DataLoader( #TODO: Make it use test subset provided by super class. 
        dataset,
        batch_size=36,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
    )

    # Test the train loader
    support, s_lbls, query, q_lbls = next(iter(train_loader))
    print(f'Support shape: {support.shape}, Labels: {s_lbls.shape}')
    print(f'Query shape: {query.shape}, Labels: {q_lbls.shape}')

    # test the test loader
    test_loader.dataset.k_shot = 1
    test_loader.dataset.validation_dataset = True
    query, q_lbls = next(iter(test_loader))
    print(f'Query shape: {query.shape}, Labels: {q_lbls.shape}')
    print()

    IMAGE_DIR = '/datasets/glomerulus-split'
    data = TripletDataset(IMAGE_DIR, TRANSFORMS, class_mapping=CUSTOM_MAPPING)
    print('Number of samples:', len(data))
    data_loader = DataLoader(
        data, batch_size=32, shuffle=True
    )
    for batch in data_loader:
        anchor, positive, negative = batch
        print(
            'Anchor shape:',
            anchor.shape,
            'Positive shape:',
            positive.shape,
            'Negative shape:',
            negative.shape,
        )
        break
def test_triplet_dataset():
    IMAGE_DIR = '/datasets/glomerulus-split'
    data = TripletDataset(IMAGE_DIR, TRANSFORMS, class_mapping=CUSTOM_MAPPING)
    print('Number of samples:', len(data))
    data_loader = DataLoader(
        data, batch_size=32, shuffle=True
    )
    for batch in data_loader:
        anchor, positive, negative = batch
        print(
            'Anchor shape:',
            anchor.shape,
            'Positive shape:',
            positive.shape,
            'Negative shape:',
            negative.shape,
        )
        break

def test_constrative_dataset():
    IMAGE_DIR = '/datasets/glomerulus-split'
    data = ContrastiveDataset(IMAGE_DIR, transform=TRANSFORMS, class_mapping=CUSTOM_MAPPING)
    print('Number of samples:', len(data))
    data_loader = DataLoader(
        data, batch_size=32, shuffle=True
    )
    for batch in data_loader:
        image, label = batch
        print(
            'Image shape:',
            image.shape,
            'Label shape:',
            label.shape,
        )
        break
   