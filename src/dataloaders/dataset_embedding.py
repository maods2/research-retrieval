import os
from typing import Any, Callable

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import numpy as np
from matplotlib.pylab import Generator
import pathology_foundation_models as pfm

from dataloaders.dataset import StandardImageDataset
from utils.auth_utils import get_hf_token
from utils.embedding_utils import load_embeddings, load_or_create_embeddings

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        config: dict[str, Any],
        return_one_hot: bool = False,
        transform: Callable[..., Any] | None = None, 
        class_mapping: dict[str, int] | None = None, 
        test_split: float = .0,
        val_split: float = .0,
        shuffle_generator: Generator | None = None,
        device: str = "cuda",
        dry_run: bool = False,
):
        super().__init__()
        assert os.path.exists(root_dir), f"Received invalid directory: {root_dir}"
        
        self.test_split = test_split or .0
        self.val_split = val_split or .0
        self.return_one_hot = return_one_hot or False
        self.transform = transform
        self.class_mapping = class_mapping
        self.shuffle_generator = shuffle_generator

        self.terumo_mlp_branches = None
        if config['model']['model_code'].endswith('branch_mlp'):
            self.terumo_mlp_branches = config['model']['n_mlp_branches']

        self.embeddings = None
        self.labels = None
        
        self.train_idxs = []
        self.test_idxs = []
        self.val_idxs = []
        #pickle_fpath = os.path.join('./data/', root_dir.split('/')[-2] + '-' + root_dir.split('/')[-1] + '-' + config['model']['model_name'] + ".pkl")

        self.embeddings, self.labels = self._load_embeddings(root_dir, config)
        #elif root_dir.endswith('pkl') or os.path.exists(pickle_fpath):
        #    self.data = pfm.dataset.EmbeddingCache.load_from_file(pickle_fpath, device=device)
        #else:
        #    assert "embedding_model" in config['data'].keys(), \
        #    """No pickle file found for this dataset. Please specify an embedding model in your configuration file under `data.embedding_model` to produce such file. 
        #    Remember to also set `data.extraction_batch_size` and `data.extraction_num_workers`. If not set, will use the values set to `data.batch_size_train` and
        #    `data.num_workers`, respectively.
        #    """
        #    self._create_embeddings(root_dir, pickle_fpath, config, device, dry_run)

        assert self.embeddings is not None and self.labels is not None, "Unreachable."

        if test_split or val_split:
            self._create_subsets()

    def _create_subsets(self):
        # list of indices. Made this way to avoid messing with positioning
        # in multiple lists at once. This way we can easily retrieve images,
        # labels, label_str and one_hot from their indices.
        idxs = np.arange(len(self.embeddings))

        if self.shuffle_generator is not None:
            self.shuffle_generator.shuffle(idxs)
        
        if self.test_split > .0:
            train_split = int(np.floor((1-self.test_split)*idxs.size))
            self.train_idxs, self.test_idxs = idxs[:train_split], idxs[train_split:]

        if self.val_split > .0:
            train_split = int(np.floor((1-self.val_split)*self.train_idxs.size))
            self.train_idxs, self.val_idxs = self.train_idxs[:train_split], self.train_idxs[train_split:]

    def _create_embeddings_pfm(self, root_dir, pickle_fpath, config, device, dry_run):
            batch_size = config['data'][
                'extraction_batch_size'
                if 'extraction_batch_size' in config['data'].keys() 
                else 'batch_size_train'
            ]
            num_workers = config['data'][
                'extraction_num_workers'
                if 'extraction_num_workers' in config['data'].keys()
                else 'num_workers'
            ]

            embedding_model = pfm.models.load_foundation_model( # TODO: support src.factories.model_factory.get_model?
                model_type=config['data']["embedding_model"],
                device=device,
                token=get_hf_token()
            )

            image_dataset = ImageFolder(
                root = root_dir, 
                transform = T.Compose([
                    T.Resize(config['transform']['train']['resize']),
                    T.ToTensor()
                ])
            )

            if dry_run:
                return

            dataset = pfm.dataset.EmbeddingCache.init_from_image_dataset(
                image_dataset=image_dataset,
                model=embedding_model,
                batch_size=batch_size,
                num_workers=num_workers,
                display_progress=True   
            )

            self.embeddings = dataset.embeddings
            self.labels =  dataset.labels

    def _load_embeddings(self, root_dir: str, config: dict[str, Any]):
        if root_dir.endswith('npz'):
            compressed_db, _ = load_embeddings(root_dir)
            return compressed_db.get('db_embeddings'), compressed_db.get('db_labels')
        else:
            raise ValueError(f"Embedding dataset file not supported: `{os.path.basename(root_dir)}`")

    def __len__(self):
        assert self.embeddings is not None, "Unreachable: `self.embeddings` is not initalized."
        return self.embeddings.__len__()

    def __getitem__(self, idx):
        assert self.embeddings is not None, "Unreachable: `self.data` is not initialized."
        return (self.embeddings.__getitem__(idx), self.labels.__getitem__(idx))

    def set_transform(self, transform):
        """
        ovwerride the base class method to set a new transform.
        """
        self.transform = transform