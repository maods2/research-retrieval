from core.base_metric import BaseMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict
from typing import List

import importlib


def create_metric_instance(metric_type: str, metric_config: Dict):
    """
    Dynamically imports and creates an instance of the specified metric class.

    Args:
        metric_type (str): Type of the metric to be created.
        metric_config (Dict): Configuration parameters for the metric.

    Returns:
        object: An instance of the specified metric class.
    """
    # Define the mapping of metric types to their respective module paths
    metric_modules = {
        'map@k': 'metrics.map_at_k.MapAtK',
    }

    if metric_type not in metric_modules:
        raise ValueError(
            f"evaluation metric '{metric_type}' is not supported."
        )

    # Dynamically import the metric class
    module_path, class_name = metric_modules[metric_type].rsplit('.', 1)
    MetricClass = getattr(importlib.import_module(module_path), class_name)

    # Define similariry function used for retrieval
    sim_functions = {
        'cosine': (cosine_similarity, 'similarity'),
        'euclidean': (euclidean_distances, 'distance'),
        'custom': (None, 'similarity'),
    }

    # Initialize the metric instance, passing additional parameters if required
    return MetricClass(**metric_config)


def get_metrics(evaluation_config: Dict) -> List[BaseMetric]:
    """
    Constructs test pipelines based on the provided evaluation configuration.

    Args:
        evaluation_config (Dict): Configuration containing a list of metrics and their parameters.

    Returns:
        List: A list of metric pipeline instances.
    """
    metrics = []

    for metric_dict in evaluation_config['list_of_metrics']:
        metric_type = metric_dict['type']
        metric_instance = create_metric_instance(metric_type, metric_dict)
        metrics.append(metric_instance)

    return metrics
