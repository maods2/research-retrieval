from abc import ABC
from abc import abstractmethod

from schemas.evaluation_context import EvaluationContext



class BaseEvaluator(ABC):
    @abstractmethod
    def test(self, ctx: EvaluationContext):
        pass

    def __call__(self, ctx: EvaluationContext):
        """
        This method allows the pipeline to be called like a function."""
        self.test(ctx)
