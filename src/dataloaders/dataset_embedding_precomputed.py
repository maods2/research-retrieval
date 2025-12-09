import numpy as np
import os
import torch

from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional, List, Any


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_mapping: Optional[dict[str, int]] = None,
        paths: Optional[List[str]] = None,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.image_paths = paths
        self.class_mapping = class_mapping

    @classmethod
    def from_npz(
        cls,
        npz_path: str | os.PathLike[str] | Path,
        class_mapping: Optional[dict[str, int]] = None,
        *,
        device: str = "cpu",
    ) -> "PrecomputedEmbeddingDataset":
        """
        Alternate constructor that skips on-the-fly embedding extraction and
        loads precomputed embeddings + labels from a .npz file.

        Expected schema of npz file:
        - "embeddings":  np.ndarray of shape (N, D)
        - "labels":      np.ndarray of shape (N,)
        - "class_mapping": optional dict mapping class names to integer labels
        - "paths": optional list/array of image paths of shape (N,)

        Optionally reads 'class_mapping' if present.
        - This bypasses __init__ and does not require root_dir, transform, or config.
        """
        path = Path(npz_path)
        npz = np.load(str(path), allow_pickle=True)

        # Minimal state needed by __len__/__getitem__
        embeddings = torch.from_numpy(npz["embeddings"]).to(device)
        labels = torch.from_numpy(npz["labels"]).to(device)
        paths = npz.get("paths", None)
        return cls(
            embeddings=embeddings,
            labels=labels,
            class_mapping=class_mapping,
            paths=paths.tolist() if paths is not None else None,
        )

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    def set_transform(self, transform: Callable[..., Any] | None):
        return