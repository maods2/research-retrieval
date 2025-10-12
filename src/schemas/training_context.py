from dataclasses import dataclass
from typing import Any, Callable

from torch.utils.data import DataLoader

from core.base_metric_logger import BaseMetricLogger

@dataclass
class TrainingContext:
    logger: Any
    metric_logger: BaseMetricLogger
    model: Any
    loss_fn: Any
    optimizer: Any
    train_loader: DataLoader
    eval_loader: DataLoader
    train_fn: Callable
    eval_fn: Callable
    metrics: list
    config: Any