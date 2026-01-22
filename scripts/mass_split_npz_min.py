#!/usr/bin/env python3
# Minimal one-off script to split a combined retrieval NPZ into train (db_*) and test (query_*) NPZs.
# Edit INPUT below and run:  python scripts/split_npz_min.py

import numpy as np
from pathlib import Path
import sys, os

# === EDIT THIS ===
#INPUT = "local_experiments/uni_fsl/glomerulus/uni_fsl_glomerulus_2025-10-18_17-33-36/embeddings_2025-10-18_17-44-06.npz"  # path to your combined file
#INPUT =  sys.argv[1]
# ================

INPUT =  "local_experiments/"
in_path = Path(INPUT)
stem = in_path.stem
TEMPLATE_OUT_TRAIN = INPUT + "{}/{}/embeddings_train.npz"
TEMPLATE_OUT_TEST = INPUT + "{}/{}/embeddings_test.npz"

def mass_split(in_path: str | Path):
    embedding_files_to_process = gather_embedding_files(in_path)

    for to_process in embedding_files_to_process:
        model_name = to_process['model_name']
        dataset_name = to_process['dataset_name']
        fpath = to_process['path']

        split_npz(
            in_path=fpath,
            out_train=TEMPLATE_OUT_TRAIN.format(model_name, dataset_name),
            out_test=TEMPLATE_OUT_TEST.format(model_name, dataset_name)
        )
        
def gather_embedding_files(in_path: str | Path) -> list[dict]:
    embedding_files_to_process = []
    for model_name in os.listdir(in_path):
        model_dir = in_path / Path(model_name)
        if not os.path.isdir(model_dir):
            continue

        for dataset_name in os.listdir(model_dir):
            model_dataset_dir = model_dir / Path(dataset_name)
            if not os.path.isdir(model_dataset_dir):
                continue

            latest_embedding_file = get_latest_embeddings_file(model_dataset_dir)
            if latest_embedding_file is None:
                continue

            latest_embedding_fpath = latest_embedding_file

            embedding_files_to_process.append({
                "model_name": model_name,
                "dataset_name": dataset_name,
                "path": latest_embedding_fpath
            })

    return embedding_files_to_process

def get_latest_embeddings_file(model_dataset_dir: str | Path):
    latest_exp_path = Path(sorted(os.listdir(model_dataset_dir))[-1])
    for fname in os.listdir(model_dataset_dir / latest_exp_path):
        if fname.startswith("embeddings_"):
            return model_dataset_dir / latest_exp_path / Path(fname)
    return None

def split_npz(in_path, out_train, out_test):
    f = np.load(str(in_path), allow_pickle=True)
    keys = list(f.files)

    # Build train payload
    train = {
        "embeddings": f["db_embeddings"],
        "labels": f["db_labels"],
    }
    if "db_paths" in keys:
        train["paths"] = f["db_paths"]
    elif "db_path" in keys:
        train["paths"] = f["db_path"]
    if "class_mapping" in keys:
        train["class_mapping"] = f["class_mapping"]

    # Build test payload
    test = {
        "embeddings": f["query_embeddings"],
        "labels": f["query_labels"],
    }
    if "query_paths" in keys:
        test["paths"] = f["query_paths"]
    if "query_classes" in keys:
        test["classes"] = f["query_classes"]
    if "class_mapping" in keys:
        test["class_mapping"] = f["class_mapping"]

    np.savez_compressed(str(out_train), **train)
    np.savez_compressed(str(out_test), **test)

    print("Done.")
    print("Train file:", out_train)
    print("Test  file:", out_test)


if __name__ == "__main__":
    mass_split(in_path)