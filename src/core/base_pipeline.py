from abc import ABC
from abc import abstractmethod


class BasePipeline(ABC):
    @abstractmethod
    def test(self, config):
        pass
