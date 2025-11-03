import os
import pathology_foundation_models as pfm
import torchvision.transforms as T
import torch
import albumentations as A
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from typing import Any, Callable

from utils.auth_utils import get_hf_token

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        config: dict[str, Any],
        transform: A.Compose,
        class_mapping: dict[str, int] | None = None, 
        # shuffle_generator: Generator | None = None,
        device: str = "cuda",
    ):
        """
        Expects path to image folder organized in the same way as expected by torchvision.datasets.ImageFolder,
        and foundation model name.
        
        Runs foundation model to generate embeddings for all images in the folder,
        and stores them in memory for fast retrieval.
        """
        super().__init__()
        assert os.path.exists(root_dir), f"Received invalid directory: {root_dir}"

        # HACK to integrate albumentations with torchvision. This is kind of bad
        self.transform = T.Lambda(lambda img: transform(image=np.array(img))['image'])
        self._full_dataset = ImageFolder(root=root_dir, transform=self.transform)

        # TODO make schema more explicit to avoid direct dict access?
        self.foundation_model = config['data']["embedding_model"]
        self.class_mapping = class_mapping
        # self.shuffle_generator = shuffle_generator
        self.device = device
        self.generate_embeddings(config)

    def generate_embeddings(self, config):
        batch_size = (
            config['data'].get('extraction_batch_size')
            or config['data']['batch_size_train']
        )
        num_workers = config['data'].get('num_workers') or 0
        embedding_model = pfm.models.load_foundation_model(  # TODO: support src.factories.model_factory.get_model?
            model_type=config['data']["embedding_model"],
            device=self.device,
            token=get_hf_token()
        )
        dataset = pfm.dataset.EmbeddingCache.init_from_image_dataset(
            image_dataset=self._full_dataset,
            model=embedding_model,
            batch_size=batch_size,
            num_workers=num_workers,
            display_progress=True   
        )
        # move to cpu to be more generally available + compatibility with pin_memory options
        self.embeddings = dataset.embeddings.to("cpu")
        self.labels = dataset.labels.to("cpu")

    # def _create_subsets(self, dataset: ImageFolder):
    #     # list of indices. Made this way to avoid messing with positioning
    #     # in multiple lists at once. This way we can easily retrieve images,
    #     # labels, label_str and one_hot from their indices.
    #     idxs = np.arange(len(dataset))
    #     if self.shuffle_generator is not None:
    #         self.shuffle_generator.shuffle(idxs)
        
    #     if self.test_split > .0:
    #         train_split = int(np.floor((1-self.test_split)*idxs.size))
    #         self.train_idxs, self.test_idxs = idxs[:train_split], idxs[train_split:]

    #     if self.val_split > .0:
    #         train_split = int(np.floor((1-self.val_split)*self.train_idxs.size))
    #         self.train_idxs, self.val_idxs = self.train_idxs[:train_split], self.train_idxs[train_split:]

    def __len__(self):
        return self.embeddings.__len__()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.embeddings[idx], self.labels[idx])

    def set_transform(self, transform: Callable[..., Any] | None):
        self.transform = transform
