from abc import ABC
from abc import abstractmethod
from core.base_metric_logger import BaseMetricLogger


class BaseEvaluator(ABC):
    @abstractmethod
    def test(self):
        pass

    def __call__(
        self,
        model,
        train_loader,
        test_loader,
        config,
        logger,
        metric_logger: BaseMetricLogger,
    ):
        """
        This method allows the pipeline to be called like a function."""
        self.test(
            model, train_loader, test_loader, config, logger, metric_logger
        )
