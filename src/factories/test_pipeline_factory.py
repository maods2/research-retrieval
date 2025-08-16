from pipelines.testing_pipes.default_testing import DefaultTestingPipeline
from pipelines.testing_pipes.few_shot_testing import FSLTestingPipeline
from typing import Any
from typing import Dict


# If you have a default OOP pipeline, import and use it here


def get_test_function(testing_config: Dict):
    pipeline = testing_config.get('pipeline', 'default')
    if pipeline == 'default':

        def default_test_fn(
            model, train_loader, test_loader, config, logger, metric_logger
        ):
            pipeline = DefaultTestingPipeline(
                model, train_loader, test_loader, config, logger, metric_logger
            )
            return pipeline.test()

        return default_test_fn
    if pipeline == 'fsl':

        def fsl_test_fn(
            model, train_loader, test_loader, config, logger, metric_logger
        ):
            pipeline = FSLTestingPipeline(
                model, train_loader, test_loader, config, logger, metric_logger
            )
            return pipeline.test()

        return fsl_test_fn
    else:
        raise ValueError(f'Testing pipeline {pipeline} is not supported')
