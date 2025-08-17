# Retrieval Research Framework

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

To extend the framework, you can add new models, loss functions, datasets, metrics (retrieval or classification), optimizers, and both training and evaluation pipelines. This typically involves implementing your custom script and registering the new function or class in the appropriate factory module.

After implementing your component, update or create configuration templates as needed. You may need to generate new dataset, model, or general config files to support your additions.

### Configuration Templates

The framework organizes experiment configuration into three template types: **dataset**, **model**, and **general**. These templates are combined to form a complete configuration for each experiment. If your model or component requires additional parameters not present in the default templates, you can create a custom template to accommodate your needs. This modular approach applies to all configurable components.

#### Generating Experiment Configurations

To generate experiment configuration files, use the `config_builder.py` script. In the `main()` function, define your experiments as a list of tuples, where each tuple specifies a combination of model and dataset parameters. For example:

```python
def main():
  # Define assets (extend as needed)
  experiments = [
    # (model_code, model_name, pipeline_type, model_template, dataset_name, dataset_template)
    ("resnet", "resnet", "default_trainer", "00-default", "skin-cancer", "skin-cancer"),
    # ("dino", "dino", "default_trainer", "00-default", "skin-cancer", "skin-cancer"),
    # ("dinov2", "dinov2", "default_trainer", "00-default", "skin-cancer", "skin-cancer"),
  ]
  # ...
```

The script determines the base configuration template automatically based on the `pipeline_type` value. If the `pipeline_type` string contains the suffix `trainer`, it sets `base_config = 'base_train'`; otherwise, it uses `base_config = 'base_test'`. This means that any pipeline type ending with `trainer` will use the training base configuration, while others (such as evaluation pipelines) will use the evaluation base configuration. This convention ensures that the correct base template is selected for each experiment type.

Edit the `experiments` list to match your desired combinations. The script will generate configuration files for each experiment, making it easy to manage and scale your research workflows.

#### Field Definitions

- **datasets**
  - `dataset_name`: Folder name used to organize configuration files related to the dataset or its variations.
  - `dataset_template`: Path to the YAML template that defines dataset-specific configuration, such as file paths, preprocessing steps, and other relevant parameters.

- **models**
  - `model_code`: Identifier used in `model_factory` to select the model.
  - `model_name`: Name for loading the model weights.
  - `pipeline_type`: Name of the training or evaluation pipeline to use (the suffix `trainer` determines if it is a training pipeline, otherwise will be evaluation pipeline).
  - `model_template`: YAML template containing model-specific configuration.

This structure ensures clarity and flexibility when defining and generating experiment configurations.


### Adding a New Model
1. Implement your model in `src/models/your_model.py`, inheriting from `BaseModel`.
2. Register your model adding it on `model_factory.py`.

### Adding a New Metric
1. Implement your metric in `src/metrics/your_metric.py`, inheriting from `BaseMetric`.
2. Register your metric in `metric_factory.py` by adding an entry to the `metric_modules` dictionary, mapping the metric name to its module path.

### Adding a New Dataset

To add a new dataset to the framework:

1. Place your dataset files in a new folder under `data/` (e.g., `data/your_dataset/`).
2. Create a dataset loader script in `src/datasets/your_dataset.py`, implementing the required dataset interface or inheriting from the appropriate base class.
3. Register your dataset in the dataset factory (e.g., `dataset_factory.py`) by adding an entry that maps the dataset name to your loader class.
4. Create a YAML configuration template for your dataset in `configs/templates/dataset/your_dataset.yaml`. This should specify dataset-specific parameters such as paths, splits, and preprocessing options.
5. Update or generate experiment configuration files to include your new dataset template as needed.

This process ensures your dataset is discoverable and configurable within the framework's modular experiment setup.


### Adding a New Training Pipeline

If the existing training pipelines do not fit your needs, you can create a custom pipeline as follows:

1. Implement your pipeline in `src/pipelines/training_pipes/your_trainer.py`, inheriting from the appropriate base class (e.g., `BaseTrainer`).
2. Integrate `metric_logger` into your pipeline to automatically log configurations and track metrics during training and evaluation.
3. Register your new pipeline in `train_factory.py` to make it discoverable by the framework.
4. Ensure your trainer implements at least the `__call__` and `train_one_epoch` methods, as these are required for the main training flow.

For reference, see the implementation in `default_trainer.py` for guidance on structuring your custom trainer.

## Environment Setup

The recommended way to use this framework is with Docker and the [VS Code Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers). The provided `Dockerfile` references a pre-built image on Docker Hub that includes PyTorch and `nvidia/cuda:12.6.3-devel-ubuntu22.04` with GPU support. If you want to review the steps used to build this image, see the `Dockerfile.toBuild` file. To open the project in a dev container, install the [VS Code Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers) and follow the prompts to reopen the folder in the container.

If you prefer not to use VS Code or dev containers, you can still use the Docker image directly. Open a terminal in your repository folder, build the container, and run it in interactive mode. Map the current directory as a volume to the container so that file changes are reflected automatically. You can then run scripts from the terminal inside the container. See the `run_gpu_container.sh` script for usage details.
```bash
# buid docker image
docker build -t retrieval-gpu-experiments .

# run docker image
docker run -it --rm --gpus all \
  --shm-size=16g \
  -v "$(pwd)":/workspaces/research-template \
  retrieval-gpu-experiments 
```

For running multiple training pipelines on a remote server, connect via SSH, use `tmux` to manage terminal sessions, build and attach the container using `run_gpu_container.sh`, download datasets and model artifacts with the provided Makefile scripts, and launch your jobs as needed.

## Running Experiments

Use the Makefile for easy experiment management:

```sh
make train CONFIG=templates/default_train.yaml
make eval CONFIG=templates/default_eval.yaml
```

Or run directly:

```sh
python3 src/main.py --config configs/templates/default_train.yaml --pipeline train
python3 src/main.py --config configs/templates/default_eval.yaml --pipeline eval
```

## Contributing
- Add docstrings to all public classes and functions.
- Follow the modular structure for new features.
- Add tests for new components in `tests/`.
