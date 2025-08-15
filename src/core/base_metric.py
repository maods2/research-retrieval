from abc import ABC
from abc import abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, targets):
        pass

    @abstractmethod
    def compute(self):
        pass
