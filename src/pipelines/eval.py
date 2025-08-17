from core.base_evaluator import BaseEvaluator
from core.base_metric_logger import BaseMetricLogger
from dataclasses import dataclass
from factories.dataset_factory import get_dataloader
from factories.evaluation_factory import get_eval_function
from factories.model_factory import get_model
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
    eval_loader: DataLoader
    eval_fn: BaseEvaluator


def setup_test_components(config) -> EvaluationContext:
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)

    transforms_eval = get_transforms(config['transform'].get('test', None))
    model = get_model(config['model'])
    train_loader, eval_loader = get_dataloader(
        config, transforms_eval, transforms_eval
    )
    eval_fn = get_eval_function(config['evaluation'])

    return EvaluationContext(
        logger=logger,
        metric_logger=metric_logger,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        eval_fn=eval_fn,
    )


def run_evaluation(ctx: EvaluationContext, config):
    ctx.logger.info('Running test...')
    ctx.eval_fn(
        ctx.model,
        ctx.train_loader,
        ctx.eval_loader,
        config,
        ctx.logger,
        ctx.metric_logger,
    )


def eval_wrapper(config):
    ctx = setup_test_components(config)
    run_evaluation(ctx, config)
