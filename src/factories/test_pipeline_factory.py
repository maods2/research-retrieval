from core.base_evaluator import BaseEvaluator
from pipelines.evaluation_pipes.default_evaluation import DefaultevaluationPipeline
from pipelines.evaluation_pipes.few_shot_evaluation import FSLevaluationPipeline
from typing import Dict


# If you have a default OOP pipeline, import and use it here


def get_eval_function(evaluation_config: Dict) -> BaseEvaluator:
    pipeline = evaluation_config.get('pipeline', 'default')
    if pipeline == 'default':
        return DefaultevaluationPipeline()
    if pipeline == 'fsl':
        return FSLevaluationPipeline()

    else:
        raise ValueError(f'evaluation pipeline {pipeline} is not supported')
