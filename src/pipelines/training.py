from dataclasses import dataclass

from factories.dataset_factory import get_dataloader
from factories.loss_factory import get_loss
from factories.model_factory import get_model
from factories.optimizer_factory import get_optimizer
from factories.test_pipeline_factory import get_test_function
from factories.train_factory import get_train_function
from factories.transform_factory import get_transforms
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


@dataclass
class PipelineContext:
    logger: any
    metric_logger: any
    model: any
    loss_fn: any
    optimizer: any
    train_loader: any
    test_loader: any
    train_fn: any
    test_fn: any


def setup_components(config)-> PipelineContext:
    """Initializes and returns all major components based on the config."""
    return PipelineContext(
    logger=setup_logger(config),
    metric_logger=setup_metric_logger(config),
    model=get_model(config['model']),
    loss_fn=get_loss(config['loss']),
    optimizer=get_optimizer(config['optimizer'], get_model(config['model'])),
    train_loader=get_dataloader(config, get_transforms(config['transform'].get('train')), get_transforms(config['transform'].get('test')))[0],
    test_loader=get_dataloader(config, get_transforms(config['transform'].get('train')), get_transforms(config['transform'].get('test')))[1],
    train_fn=get_train_function(config),
    test_fn=get_test_function(config['testing']),
    )

def run_training(ctx: PipelineContext, config):
    ctx.logger.info('Starting training...')
    ctx.train_fn(ctx.model, ctx.loss_fn, ctx.optimizer,
                 ctx.train_loader, ctx.test_loader,
                 config, ctx.logger, ctx.metric_logger)

def run_testing(ctx: PipelineContext, config):
    if config['testing'].get('enabled'):
        ctx.logger.info('Running testing...')
        ctx.test_fn(ctx.model, ctx.train_loader, ctx.test_loader,
                    config, ctx.logger, ctx.metric_logger)

def train_wrapper(config):
    ctx = setup_components(config)
    run_training(ctx, config)
    run_testing(ctx, config)
