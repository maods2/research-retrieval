from core.base_metric_logger import BaseMetricLogger
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class TrainingContext:
    logger: any
    metric_logger: BaseMetricLogger
    model: any
    loss_fn: any
    optimizer: any
    train_loader: DataLoader
    eval_loader: DataLoader
    train_fn: callable
    eval_fn: callable
    metrics: list
    config: any
