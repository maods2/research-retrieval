from core.base_evaluator import BaseEvaluator
from pipelines.evaluation_pipes.default_eval import DefaultEvaluationPipeline
from pipelines.evaluation_pipes.few_shot_eval import FSLevaluationPipeline
from typing import Dict


def get_eval_function(evaluation_config: Dict) -> BaseEvaluator:
    pipeline = evaluation_config.get('pipeline', 'default')
    if pipeline == 'default':
        return DefaultEvaluationPipeline()
    if pipeline == 'fsl':
        return FSLevaluationPipeline()

    else:
        raise ValueError(f'evaluation pipeline {pipeline} is not supported')
