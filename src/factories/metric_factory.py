from typing import Dict, List, Callable
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import importlib

from core.base_metric import BaseMetric
from factories.model_factory import get_model


# ---------------------------
# Registries
# ---------------------------
METRIC_REGISTRY = {
    "map@k": "metrics.map_at_k.MapAtK",
    # extend as needed
}


CUSTOM_SIM_REGISTRY = {
    "custom_similarity": lambda x, y: cosine_similarity(x, y),
    "retriev_learned_sim": lambda x, y: _load_learned_similarity(x, y),
    # add more custom similarity functions here
}


# ---------------------------
# Similarity function factory
# ---------------------------
def get_similarity_function(metric_config: Dict, config: Dict) -> Callable:
    """Returns a similarity function based on the config."""
    similarity_fn = metric_config.get("similarity_fn")

    if similarity_fn == "cosine":
        return cosine_similarity
    elif similarity_fn == "euclidean":
        return euclidean_distances
    elif similarity_fn in CUSTOM_SIM_REGISTRY:
        return CUSTOM_SIM_REGISTRY[similarity_fn]
    
    else:
        raise ValueError(f"Similarity function '{similarity_fn}' is not supported.")


def _load_learned_similarity(config: Dict) -> Callable:
    """Example: load learned similarity from a model checkpoint."""
    # TODO: Should be improved to indentify if checkpoint is needed or not.
    # If eval pipeline is called after training, model artifict is available in config,
    # For eval pipeline checkpoint needs to be defined in config manually.
    
    # config["model"]["load_checkpoint"] = True
    # model = get_model(config["model"])
    # return model.pairwise_similarity
    pass


# ---------------------------
# Metric factory
# ---------------------------
def create_metric_instance(metric_type: str, metric_config: Dict, config: Dict) -> BaseMetric:
    """Creates and returns a metric instance with the appropriate similarity function."""
    if metric_type not in METRIC_REGISTRY:
        raise ValueError(f"Evaluation metric '{metric_type}' is not supported.")

    # Load metric class dynamically
    module_path, class_name = METRIC_REGISTRY[metric_type].rsplit(".", 1)
    MetricClass = getattr(importlib.import_module(module_path), class_name)

    # Resolve similarity function
    sim_fn = get_similarity_function(metric_config, config)
    similarity_type = metric_config.get("similarity_type")  # 'similarity' or 'distance'
    m_conf_copy = metric_config.copy()
    m_conf_copy["similarity_fn"] = (sim_fn, similarity_type)

    return MetricClass(**m_conf_copy)


# ---------------------------
# Entry point for all metrics
# ---------------------------
def get_metrics(config: Dict) -> List[BaseMetric]:
    """Constructs all evaluation metrics from config."""
    metrics = []
    for metric_dict in config["evaluation"]["list_of_metrics"]:
        metric_type = metric_dict["type"]
        metric_instance = create_metric_instance(metric_type, metric_dict, config)
        metrics.append(metric_instance)
    return metrics
