from factories.dataset_factory import get_dataloader
from factories.evaluation_factory import get_eval_function
from factories.metric_factory import get_metrics
from factories.model_factory import get_model
from factories.transform_factory import get_transforms
from schemas.evaluation_context import EvaluationContext
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


def setup_test_components(config) -> EvaluationContext:
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)

    transforms_eval = get_transforms(config['transform'].get('test', None))
    model = get_model(config['model'])
    train_loader, eval_loader = get_dataloader(
        config, transforms_eval, transforms_eval
    )
    eval_fn = get_eval_function(config['evaluation'])
    metrics = get_metrics(config)

    return EvaluationContext(
        logger=logger,
        metric_logger=metric_logger,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        eval_fn=eval_fn,
        metrics=metrics,
        config=config,
    )


def run_evaluation(ctx: EvaluationContext):
    ctx.logger.info('Running test...')
    ctx.eval_fn(ctx)


def eval_wrapper(config):
    ctx = setup_test_components(config)
    run_evaluation(ctx)
