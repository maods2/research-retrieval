from core.base_evaluator import BaseEvaluator
from core.base_metric_logger import BaseMetricLogger
from dataclasses import dataclass
from factories.dataset_factory import get_dataloader
from factories.model_factory import get_model
from factories.test_pipeline_factory import get_test_function
from factories.transform_factory import get_transforms
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


@dataclass
class EvaluationContext:
    logger: any
    metric_logger: BaseMetricLogger
    model: any
    train_loader: DataLoader
    test_loader: DataLoader
    test_fn: BaseEvaluator


def setup_test_components(config) -> EvaluationContext:
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)

    transforms_test = get_transforms(config['transform'].get('test', None))
    model = get_model(config['model'])
    train_loader, test_loader = get_dataloader(
        config, transforms_test, transforms_test
    )
    test_fn = get_test_function(config['testing'])

    return EvaluationContext(
        logger=logger,
        metric_logger=metric_logger,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        test_fn=test_fn,
    )


def run_testing(ctx: EvaluationContext, config):
    ctx.logger.info('Running test...')
    ctx.test_fn(
        ctx.model,
        ctx.train_loader,
        ctx.test_loader,
        config,
        ctx.logger,
        ctx.metric_logger,
    )


def test_wrapper(config):
    ctx = setup_test_components(config)
    run_testing(ctx, config)