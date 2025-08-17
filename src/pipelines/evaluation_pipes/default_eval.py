from core.base_evaluator import BaseEvaluator
from core.base_metric_logger import BaseMetricLogger
from factories.metric_factory import get_metrics
from schemas.evaluation_context import EvaluationContext
from utils.embedding_utils import load_or_create_embeddings

import torch


class DefaulRetrievaltEvaluator(BaseEvaluator):
    def test(self, ctx: EvaluationContext):
        
        metrics_list = get_metrics(ctx.config['evaluation'])

        device = (
            ctx.config['device']
            if ctx.config.get('device')
            else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        embeddings = load_or_create_embeddings(
            ctx.model, ctx.train_loader, ctx.eval_loader, ctx.config, ctx.logger, device
        )
        for metric in metrics_list:
            results = metric(
                ctx.model, ctx.train_loader, ctx.eval_loader, embeddings, ctx.config, ctx.logger
            )
            ctx.logger.info(f'Results for {metric.__class__.__name__}: {results}')
            ctx.metric_logger.log_metrics(results)
