from core.base_evaluator import BaseEvaluator
from core.base_metric import MetricLoggerBase
from factories.metric_factory import get_metrics
from utils.embedding_utils import load_or_create_embeddings

import torch


class DefaultevaluationPipeline(BaseEvaluator):
    def test(
        self,
        model,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: MetricLoggerBase,
    ):
        metrics_list = get_metrics(config['evaluation'])
        config = config
        logger = logger
        metric_logger = metric_logger
        model = model
        train_loader = train_loader
        test_loader = test_loader
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

