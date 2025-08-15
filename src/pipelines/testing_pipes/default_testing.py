from core.base_pipeline import BasePipeline
from core.metric_base import MetricLoggerBase
from factories.metric_factory import get_metrics
from utils.embedding_utils import load_or_create_embeddings

import torch


class DefaultTestingPipeline(BasePipeline):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: MetricLoggerBase,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.metric_logger = metric_logger

    def test(self):
        metrics_list = get_metrics(self.config['testing'])
        config = self.config
        logger = self.logger
        metric_logger = self.metric_logger
        model = self.model
        train_loader = self.train_loader
        test_loader = self.test_loader
        device = (
            config['device']
            if config.get('device')
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        embeddings = load_or_create_embeddings(
            model, train_loader, test_loader, config, logger, device
        )
        for metric in metrics_list:
            results = metric(
                model, train_loader, test_loader, embeddings, config, logger
            )
            logger.info(f'Results for {metric.__class__.__name__}: {results}')
            metric_logger.log_metrics(results)
        return True
