
from configs.builder_utils import generate


def main():
    # Define assets (extend as needed)

    experiments = [
    #   (model_code,        model_name,         pipeline_type,          model_template    dataset_name,   dataset_template  )                         


        # Supervised Contrastive Learning (SupCon) experiments
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "glomerulo",  "glomerulo" ),
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "bracs",          "bracs" ),
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "crc-val-he-7k",  "crc-val-he-7k" ),
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "lung-colon",     "lung-colon" ),
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "skin-cancer",    "skin-cancer" ),
        ("supcon",          "resnet50",          "supcon_trainer",       "03-supcon",       "ubc-ovarian-cancer", "ubc-ovarian-cancer" ),
        
        # Supervised Hashing experiments
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "glomerulo",  "glomerulo" ),
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "bracs",          "bracs" ),
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "crc-val-he-7k",  "crc-val-he-7k" ),
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "lung-colon",     "lung-colon" ),
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "skin-cancer",    "skin-cancer" ),
        ("liu_dsh",         "liu_dsh",           "supervised_hashing_trainer", "04-supervised-hashing", "ubc-ovarian-cancer", "ubc-ovarian-cancer" ),
        
        
        # Autoencoder experiments
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "glomerulo",  "glomerulo" ),
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "bracs",          "bracs" ),
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "crc-val-he-7k",  "crc-val-he-7k" ),
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "lung-colon",     "lung-colon" ),
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "skin-cancer",    "skin-cancer" ),
        ("autoencoder",     "resnet50",          "autoencoder_trainer",  "06-autoencoder",  "ubc-ovarian-cancer", "ubc-ovarian-cancer" ),   
        
        
        # Triplet Loss experiments
        ("triplet",         "resnet50",          "triplet_trainer",        "05-triplet",      "glomerulo",  "glomerulo" ),
        ("triplet",         "resnet50",          "triplet_trainer",        "05-triplet",      "bracs",          "bracs" ),
        ("triplet",         "resnet50",          "triplet_trainer",       "05-triplet",      "crc-val-he-7k",  "crc-val-he-7k" ),
        ("triplet",         "resnet50",          "triplet_trainer",       "05-triplet",      "lung-colon",     "lung-colon" ),
        ("triplet",         "resnet50",          "triplet_trainer",       "05-triplet",      "skin-cancer",    "skin-cancer" ),
        ("triplet",         "resnet50",          "triplet_trainer",       "05-triplet",      "ubc-ovarian-cancer", "ubc-ovarian-cancer" ),
        
        
        
        
        
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
    
    # Generate regular configurations
    print("Generating regular configurations...")
    generate(experiments, test_mode=False)
    
    # Generate test configurations
    print("\nGenerating test configurations...")
    generate(experiments, test_mode=True)


if __name__ == '__main__':
    main()
