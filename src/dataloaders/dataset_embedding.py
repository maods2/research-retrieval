import os
from typing import Any, Callable

import torch
from torchvision.datasets import ImageFolder

from matplotlib.pylab import Generator
import pathology_foundation_models as pfm

from src.dataloaders import StandardImageDataset


class DatasetEmbedding(StandardImageDataset):
    def __init__(
        self,
        root_dir: str,
        embedding_model: torch.nn.Module | pfm.models.FoundationModel,
        batch_size: int,
        return_one_hot: bool = False,
        transform: Callable[..., Any] | None = None, 
        config: dict[str, Any] | None = None, 
        class_mapping: dict[str, int] | None = None, 
        test_split: float | None = None, 
        val_split: float | None = None, 
        shuffle_generator: Generator | None = None,
        num_workers: int = 4,
        device: str = "cuda"
):
        super().__init__(
            root_dir, 
            return_one_hot, 
            transform, 
            config, 
            class_mapping, 
            test_split, 
            val_split, 
            shuffle_generator
        )
        
        assert os.path.exists(root_dir), f"Received invalid directory: {root_dir}"
        
        dataset_fpath = os.path.join(root_dir, root_dir.split('/')[-1])
        self.data = None

        if os.path.exists(dataset_fpath):
            self.data = pfm.dataset.EmbeddingCache.load_from_file(dataset_fpath, device=device)
        else:
            image_dataset = ImageFolder(root_dir, transform=transform)

            if not isinstance(embedding_model, pfm.models.FoundationModel):
                embedding_model.__dict__['device'] = device # HACK: This is so we can utilize nn.Modules over PFM's methods

            self.data = pfm.dataset.EmbeddingCache.init_from_image_dataset(
                image_dataset=image_dataset,
                model=embedding_model,
                batch_size=batch_size,
                num_workers=num_workers,
                display_progress=True   
            )
            self.data.save(dataset_fpath)

        assert self.data is not None, "Unreachable."

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)