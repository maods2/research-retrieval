# Research Template Framework

## Overview

This framework is designed for modular research in deep learning, supporting multiple models, metrics, and training/evaluation pipelines. It is extensible and easy to use for rapid experimentation.

## Architecture

- **src/core/**: Base abstract classes for models, metrics, and pipelines.
- **src/models/**: Model implementations and registration.
- **src/metrics/**: Metric implementations and registration.
- **src/pipelines/**: Training and evaluation pipeline implementations.
- **src/utils/**: Utility functions and config loader.
- **configs/**: YAML configuration files for experiments.

## Extensibility

### Adding a New Model
1. Implement your model in `src/models/your_model.py`, inheriting from `BaseModel`.
2. Register it using the `@register_model('your_model_name')` decorator from `model_factory.py`.

### Adding a New Metric
1. Implement your metric in `src/metrics/your_metric.py`, inheriting from `BaseMetric`.
2. Register it using the `@register_metric('your_metric_name')` decorator from `metric_factory.py`.

### Adding a New Pipeline
1. Implement your pipeline in `src/pipelines/your_pipeline.py`, inheriting from `BasePipeline`.
2. Register it similarly if using a factory.

## Example Config

See `configs/templates/few_shot/default_train_config.yaml` for a full example. The config controls model, pipeline, metrics, and training parameters.

## Running Experiments

Use the Makefile for easy experiment management:

```sh
make train CONFIG=templates/few_shot/default_train_config.yaml
make test CONFIG=templates/few_shot/default_train_config.yaml
```

Or run directly:

```sh
python src/main.py --config configs/templates/few_shot/default_train_config.yaml --pipeline train
```

## Contributing
- Add docstrings to all public classes and functions.
- Follow the modular structure for new features.
- Add tests for new components in `tests/`.
