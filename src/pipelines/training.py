from factories.dataset_factory import get_dataloader
from factories.loss_factory import get_loss
from factories.model_factory import get_model
from factories.optimizer_factory import get_optimizer
from factories.test_pipeline_factory import get_test_function
from factories.train_factory import get_train_function
from factories.transform_factory import get_transforms
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


def setup_components(config):
    """Initializes and returns all major components based on the config."""
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)
    model = get_model(config['model'])
    loss_fn = get_loss(config['loss'])
    optimizer = get_optimizer(config['optimizer'], model)
    transforms_train = get_transforms(config['transform'].get('train'))
    transforms_test = get_transforms(config['transform'].get('test'))
    train_loader, test_loader = get_dataloader(
        config, transforms_train, transforms_test
    )
    train_fn = get_train_function(config)
    test_fn = get_test_function(config['testing'])
    return (
        logger,
        metric_logger,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        train_fn,
        test_fn,
    )


def run_training(
    train_fn,
    model,
    loss_fn,
    optimizer,
    train_loader,
    test_loader,
    config,
    logger,
    metric_logger,
):
    """Runs the training loop."""
    logger.info('Starting training...')
    train_fn(
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    )


def run_testing(
    test_fn, model, train_loader, test_loader, config, logger, metric_logger
):
    """Runs the testing loop if enabled."""
    if config['testing'].get('enabled'):
        logger.info('Running testing...')
        test_fn(
            model, train_loader, test_loader, config, logger, metric_logger
        )


def train_wrapper(config):
    """
    Orchestrates the training and testing pipeline.
    """
    (
        logger,
        metric_logger,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        train_fn,
        test_fn,
    ) = setup_components(config)
    run_training(
        train_fn,
        model,
        loss_fn,
        optimizer,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    )
    run_testing(
        test_fn,
        model,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger,
    )
