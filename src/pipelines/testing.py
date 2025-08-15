from factories.dataset_factory import get_dataloader
from factories.model_factory import get_model
from factories.test_pipeline_factory import get_test_function
from factories.transform_factory import get_transforms
from utils.logger import setup_logger
from utils.metric_logger import setup_metric_logger


def test_wrapper(config):
    logger = setup_logger(config)
    metric_logger = setup_metric_logger(config)

    # load model
    transforms_test = get_transforms(config['transform'].get('test', None))

    model = get_model(config['model'])

    train_loader, test_loader = get_dataloader(
        config, transforms_test, transforms_test
    )

    # Função de teste
    test_fn = get_test_function(config['testing'])

    # Teste
    logger.info('Running test...')
    test_fn(model, train_loader, test_loader, config, logger, metric_logger)
