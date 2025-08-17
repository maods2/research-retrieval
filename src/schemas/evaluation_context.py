from core.base_metric_logger import BaseMetricLogger
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class EvaluationContext:
    logger: any
    metric_logger: BaseMetricLogger
    model: any
    train_loader: DataLoader
    eval_loader: DataLoader
    eval_fn: callable
    config: any
