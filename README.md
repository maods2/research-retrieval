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

## Using the Framework
1. Generating configs
2. Runnning the experiments
3. Results


## Extensibility

To extend the framework, you can add new models, loss functions, datasets, metrics (retrieval or classification), optimizers, and both training and evaluation pipelines. Typically, this involves creating your custom script and registering the new function or class in the appropriate factory module.

After implementing your component, update or create configuration templates as needed. You may need to generate new dataset, model, or general config files to support your additions.

The framework uses three types of config: templates—dataset, model, and general—which together form a complete experiment configuration. If your model or component requires extra parameters not present in the default templates, you can create a new template specific to your needs. This approach applies to all configurable components. 


To generate experiment configuration files, use the `config_template_builder.py` script. In the `main()` function, specify the combinations of datasets, models, and training pipelines you want to generate configs for. For example:

```python
# ...
def main():
    # Define datasets (add or remove as needed)
    datasets = [
        # (dataset_name, dataset_template)
        ('skin-cancer', 'skin-cancer'),
    ]

    # Define models (model_code, model_name, pipeline_type, model_template)
    models = [
        ("resnet", "resnet", "default_trainer", "00-default"),
        ("dino", "dino", "default_trainer", "00-default"),
        # ...
    ]

    # ...
```

Edit the `datasets` and `models` lists to match your experiment needs. The script will generate configuration files for all specified combinations, making it easy to scale and manage experiments.

### Adding a New Model
1. Implement your model in `src/models/your_model.py`, inheriting from `BaseModel`.
2. Register your model adding it on `model_factory.py`.

### Adding a New Metric
1. Implement your metric in `src/metrics/your_metric.py`, inheriting from `BaseMetric`.
2. Register your metric in `metric_factory.py` by adding an entry to the `metric_modules` dictionary, mapping the metric name to its module path.

### Adding New dataset


### Adding a New Training Pipeline

If the existing training pipelines do not fit your needs, you can create a custom pipeline as follows:

1. Implement your pipeline in `src/pipelines/your_trainer.py`, inheriting from the appropriate base class (e.g., `BaseTrainer`).
2. Integrate `metric_logger` into your pipeline to automatically log configurations and track metrics during training and evaluation.
3. Register your new pipeline in `train_factory.py` to make it discoverable by the framework.
4. Ensure your trainer implements at least the `__call__` and `train_one_epoch` methods, as these are required for the main training flow.

For reference, see the implementation in `default_trainer.py` for guidance on structuring your custom trainer.




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
