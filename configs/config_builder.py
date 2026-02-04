import yaml
from pathlib import Path
from copy import deepcopy
from itertools import product
from ruamel.yaml import YAML

ATT_METRIC = True # false is SEB

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
        
        base_config = 'base_train' if 'trainer' in pipeline_type else 'base_eval'
        
        base_path = Path(f'configs/templates/general/{base_config}.yml')
        data_path = Path(f'configs/templates/datasets/{dataset_template}.yml')
        out_dir = Path(f'test_configs/{"att_metric" if ATT_METRIC else "seb"}/{dataset_name}')
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
    #   (model_code,        model_name,         pipeline_type,          model_template    dataset_name,   dataset_template  )                         

        #("resnet",          "resnet18_classif",  "default_trainer",      "02-resnet-clsf",    "ovarian-cancer",  "ovarian-cancer" ),
        #("resnet_fsl",      "resnet18",          "fsl_trainer",          "01-few-shot",      "ovarian-cancer",  "ovarian-cancer-fsl" ),
        #("resnet_fsl",      "resnet18",          "retrieval_evaluator",  "01-few-shot",      "ovarian-cancer-fsl-eval",  "ovarian-cancer-fsl" ),
        
        
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

    # =========  Semantic Embedding Builder  ======== 
    models_seb = [
        # model_code        model_name       model_templates
        ("uni_fsl",             "uni",           "03-seb-fsl"),
        ("uni2h_fsl",           "uni2h",         "03-seb-fsl"),
        ("virchow_fsl",         "virchow",       "03-seb-fsl"),
        ("virchow_v2_fsl",      "virchow_v2",    "03-seb-fsl"),
        ("phikon_fsl",          "phikon",        "03-seb-fsl"),
        ("phikon_v2_fsl",       "phikon_v2",     "03-seb-fsl"),
        ("dino_fsl",            "dino_v1_b16",   "03-seb-fsl"),
        ("dinov2_fsl",          "dino_v2_b",     "03-seb-fsl"),
        ("dinov3_fsl",          "dinov3",        "03-seb-fsl"),
    ]
    # ==============  Attention Metric  =============
    models_att_metric = [
        ("uni",             "uni",           "04-att-metric"),
        ("uni2h",           "uni2h",         "04-att-metric"),
        ("virchow",         "virchow",       "04-att-metric"),
        ("virchow_v2",      "virchow_v2",    "04-att-metric"),
        ("phikon",          "phikon",        "04-att-metric"),
        ("phikon_v2",       "phikon_v2",     "04-att-metric"),
        ("dino_v1_b16",     "dino_v1_b16",   "04-att-metric"),
        ("dino_v2_b",       "dino_v2_b",     "04-att-metric"),
    ]

    # =========  Semantic Embedding Builder  ======== 
    datasets_seb = [
    #   dataset_name   dataset_template
        ("glomerulo", "glomerulo-fsl"),
        ("bracs", "bracs-fsl"),
        ("crc-val-he-7k", "crc-val-he-7k-fsl"),
        ("lung-colon", "lung-colon-fsl"),
        ("skin-cancer", "skin-cancer-fsl"),
        ("ubc-ovarian-cancer", "ubc-ovarian-cancer-fsl"),
    ]
    # ==============  Attention Metric  =============
    datasets_att_metric = [
        ("glomerulo",          "precomputed-glomerulo"),
        ("bracs",              "precomputed-bracs"),
        ("crc-val-he-7k",      "precomputed-crc-val-he-7k"),
        ("lung-colon",         "precomputed-lung-colon"),
        ("skin-cancer",        "precomputed-skin-cancer"),
        ("ubc-ovarian-cancer", "precomputed-ubc-ovarian-cancer"),
    ]
    
    # =========  Semantic Embedding Builder  ======== 
    pipeline_seb =  [
        "fsl_trainer"
    ]

    # ==============  Attention Metric  =============
    pipeline_att_metric = [
        "terumo_trainer"
    ]

    
    for exp in product(
        models_att_metric   if ATT_METRIC else models_seb, 
        datasets_att_metric if ATT_METRIC else datasets_seb, 
        pipeline_att_metric if ATT_METRIC else pipeline_seb
    ):
        model_name, model_code, model_template = exp[0]
        dataset_name, dataset_template = exp[1]
        pipeline = exp[2]
        experiments.append((model_name, model_code, pipeline, model_template, dataset_name, dataset_template))
                            
    # Generate configurations
    generate(experiments)


if __name__ == '__main__':
    main()
