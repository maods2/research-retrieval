from core.base_evaluator import BaseEvaluator
from pipelines.evaluation_pipes.default_eval import DefaulRetrievaltEvaluator
from pipelines.evaluation_pipes.few_shot_eval import FSLEvaluator
from typing import Dict


def get_eval_function(evaluation_config: Dict) -> BaseEvaluator:
    pipeline = evaluation_config.get('pipeline', 'default')
    if pipeline == 'retrieval_evaluator':
        return DefaulRetrievaltEvaluator()
    if pipeline == 'fsl_evaluator':
        return FSLEvaluator()

    else:
        raise ValueError(f'evaluation pipeline {pipeline} is not supported')
