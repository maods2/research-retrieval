from abc import ABC
from abc import abstractmethod


class MetricBase(ABC):
    """
    Base interface for all metrics. All metrics must inherit from this class and implement the required methods.
    """

    @abstractmethod
    def __call__(
        self, model, train_loader, test_loader, embeddings, config, logger
    ):
        """
        Method to compute the metric.

        Args:
            model: PyTorch model being tested.
            train_loader: DataLoader for the training data.
            test_loader: DataLoader for the evaluation data.
            embeddings: Dictionary containing embeddings.
            config: Configuration dictionary.
            logger: Logger object for logging.

        Returns:
            dict: Metric results in the form of a dictionary (e.g., {"metric_name": value}).
        """
        raise NotImplementedError('Subclasses must implement this method.')
