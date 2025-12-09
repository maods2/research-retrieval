#!/usr/bin/env python3
# Minimal one-off script to split a combined retrieval NPZ into train (db_*) and test (query_*) NPZs.
# Edit INPUT below and run:  python scripts/split_npz_min.py

import numpy as np
from pathlib import Path
import sys

# === EDIT THIS ===
#INPUT = "local_experiments/uni_fsl/glomerulus/uni_fsl_glomerulus_2025-10-18_17-33-36/embeddings_2025-10-18_17-44-06.npz"  # path to your combined file
INPUT =  sys.argv[1]
# ================

in_path = Path(INPUT)
stem = in_path.stem
out_train = Path("local_experiments/uni_fsl/glomerulus/embeddings_train.npz")
out_test = Path("local_experiments/uni_fsl/glomerulus/embeddings_test.npz")

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
