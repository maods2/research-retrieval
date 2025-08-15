from abc import ABC
from abc import abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass
