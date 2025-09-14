import os
from typing import Any, Callable

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from matplotlib.pylab import Generator
import pathology_foundation_models as pfm

from dataloaders.dataset import StandardImageDataset
from utils.auth_utils import get_hf_token

class EmbeddingDataset(StandardImageDataset):
    def __init__(
        self,
        root_dir: str,
        config: dict[str, Any],
        return_one_hot: bool = False,
        transform: Callable[..., Any] | None = None, 
        class_mapping: dict[str, int] | None = None, 
        test_split: float | None = None, 
        val_split: float | None = None, 
        shuffle_generator: Generator | None = None,
        device: str = "cuda",
        dry_run: bool = False,
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
        
        pickle_fpath = os.path.join('./data/', root_dir.split('/')[-1] + config['model']['model_name'] + ".pkl")
        self.data = None

        if os.path.exists(pickle_fpath):
            self.data = pfm.dataset.EmbeddingCache.load_from_file(pickle_fpath, device=device)
        else:
            assert "embedding_model" in config['data'].keys(), "No pickle file found for this dataset.\
                Please specify an embedding model in your configuration file under `data.embedding_model` to produce such file.\
                Remember to also set `data.extraction_batch_size` and `data.extraction_num_workers`. If not set, will use the values \
                set to `data.batch_size_train` and `data.num_workers`, respectively."

            batch_size = config['data']['extraction_batch_size'] \
                            if 'extraction_batch_size' in config['data'].keys() \
                            else config['data']['batch_size_train']

            num_workers = config['data']['extraction_num_workers'] \
                            if 'extraction_num_workers' in config['data'].keys() \
                            else config['data']['num_workers']

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

            self.data = pfm.dataset.EmbeddingCache.init_from_image_dataset(
                image_dataset=image_dataset,
                model=embedding_model,
                batch_size=batch_size,
                num_workers=num_workers,
                display_progress=True   
            )
            self.data.labels.to(device)
            self.data.save(pickle_fpath)

        assert self.data is not None, "Unreachable."

    def __len__(self):
        assert self.data is not None, "Unreachable: `self.data` is not initalized."
        return self.data.__len__()

    def __getitem__(self, idx):
        assert self.data is not None, "Unreachable: `self.data` is not initialized."
        return self.data.__getitem__(idx)