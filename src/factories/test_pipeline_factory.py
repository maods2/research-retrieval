from core.base_evaluator import BaseEvaluator
from pipelines.testing_pipes.default_testing import DefaultTestingPipeline
from pipelines.testing_pipes.few_shot_testing import FSLTestingPipeline
from typing import Dict


# If you have a default OOP pipeline, import and use it here


def get_test_function(testing_config: Dict) -> BaseEvaluator:
    pipeline = testing_config.get('pipeline', 'default')
    if pipeline == 'default':
        return DefaultTestingPipeline()
    if pipeline == 'fsl':
        return FSLTestingPipeline()

    else:
        raise ValueError(f'Testing pipeline {pipeline} is not supported')
