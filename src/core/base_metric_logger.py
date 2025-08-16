from abc import ABC
from abc import abstractmethod


class BaseMetricLogger(ABC):
    """
    Base interface for metric logging. Allows switching between different logging backends.
    """

    @abstractmethod
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """
        Log a single metric.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """
        Log multiple metrics.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_params(self, params: dict):
        """
        Log parameters.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_artifact(self, artifact_path: str):
        """
        Log an artifact.
        """
        raise NotImplementedError('Subclasses must implement this method.')

    @abstractmethod
    def log_json(self, artifact_path: str, base_filename: str):
        """
        Log an artifact.
        """
        raise NotImplementedError('Subclasses must implement this method.')
