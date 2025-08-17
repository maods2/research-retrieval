from factories.dataset_factory import get_dataloader
from factories.evaluation_factory import get_eval_function
from factories.loss_factory import get_loss
from factories.metric_factory import get_metrics
from factories.model_factory import get_model
from factories.optimizer_factory import get_optimizer
from factories.train_factory import get_train_function
from factories.transform_factory import get_transforms
from schemas.training_context import TrainingContext
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


def setup_components(config) -> TrainingContext:
    """Initializes and returns all major components based on the config."""
    train_loader, eval_loader = get_dataloader(
        config,
        get_transforms(config['transform'].get('train')),
        get_transforms(config['transform'].get('test')),
    )
    return TrainingContext(
        logger=setup_logger(config),
        metric_logger=setup_metric_logger(config),
        model=get_model(config['model']),
        loss_fn=get_loss(config['loss']),
        optimizer=get_optimizer(
            config['optimizer'], get_model(config['model'])
        ),
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_fn=get_train_function(config),
        eval_fn=get_eval_function(config['evaluation']),
        metrics=get_metrics(config),
        config=config,
    )


def run_training(ctx: TrainingContext):
    ctx.logger.info('Starting training...')
    ctx.train_fn(ctx)


def run_evaluation(ctx: TrainingContext):
    if ctx.config['evaluation'].get('enabled'):
        ctx.logger.info('Running evaluation...')
        ctx.eval_fn(ctx)


def train_wrapper(config):
    ctx = setup_components(config)
    run_training(ctx)
    run_evaluation(ctx)
