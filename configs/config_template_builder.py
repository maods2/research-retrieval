import yaml
from pathlib import Path
from copy import deepcopy
from ruamel.yaml import YAML

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def merge_dicts(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_dicts(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base

def replace_placeholders(config, replacements):
    if isinstance(config, dict):
        return {k: replace_placeholders(v, replacements) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_placeholders(i, replacements) for i in config]
    elif isinstance(config, str):
        for key, value in replacements.items():
            config = config.replace(f'<{key}>', str(value))
        return config
    else:
        return config

def build_config(base_path, model_path, data_path, replacements):
    base = load_yaml(base_path)
    model = load_yaml(model_path)
    data = load_yaml(data_path)
    config = {}
    for cfg in [data, model, base]:
        merge_dicts(config, cfg)
    config = replace_placeholders(config, replacements)
    return config

def save_config(config, out_path):
    yaml = YAML()
    yaml.default_flow_style = None  # Use block style for objects but inline style for arrays
    with open(out_path, 'w') as f:
        yaml.dump(config, f)

def generate(experiments):
    for model_code, model_name, pipeline_type, model_template, dataset_name, dataset_template in experiments:
        
        base_config = 'base_train' if 'trainer' in pipeline_type else 'base_test'
        
        base_path = Path(f'configs/templates/general/{base_config}.yml')
        data_path = Path(f'configs/templates/datasets/{dataset_template}.yml')
        out_dir = Path(f'configs/{dataset_name}')
        out_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = Path(f'configs/templates/models/{model_template}.yml')
        replacements = {
            'pipeline_type': pipeline_type,
            'model_code': model_code,
            'dataset_name': dataset_name,
            'model_name': model_name
        }
        config = build_config(base_path, model_path, data_path, replacements)
        out_path = out_dir / f'{model_code}.yml'
        save_config(config, out_path)
        print(f'Generated: {out_path}')


def main():
    # Define assets (extend as needed)

    experiments = [
    #   (model_code,        model_name,         pipeline_type,          model_template    dataset_name,   dataset_template)                            )

        ("resnet",          "resnet18_classif",   "default_trainer",      "00-default",      "ovarian-cancer",  "ovarian-cancer" ),
        
        
        # ("dino",            "dino",             "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("dinov2",          "dinov2",           "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("vit",             "vit",              "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("uni",             "uni",              "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("UNI2-h",          "UNI2-h",           "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("virchow2",        "virchow2",         "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("phikon",          "phikon",           "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("phikon-v2",       "phikon-v2",        "default_trainer",      "00-default",      "skin-cancer",  "skin-cancer" ),
        # ("resnet_fsl",      "resnet50",         "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("dino_fsl",        "dino_fsl",         "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("dinov2_fsl",      "dinov2_fsl",       "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("vit_fsl",         "vit_fsl",          "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("uni_fsl",         "uni_fsl",          "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("UNI2-h_fsl",      "UNI2-h_fsl",       "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("virchow2_fsl",    "virchow2_fsl",     "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("phikon_fsl",      "phikon_fsl",       "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),
        # ("phikon-v2_fsl",   "phikon-v2_fsl",    "fsl_trainer",          "01-few-shot",     "skin-cancer",  "skin-cancer" ),

    ]
    
    # Generate configurations
    generate(experiments)


if __name__ == '__main__':
    main()
