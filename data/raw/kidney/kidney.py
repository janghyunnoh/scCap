# ============================================
# Kidney scRNA-seq preprocessing (raw version, no filtering)
# From: https://cellxgene.cziscience.com/collections/0f528c8a-a25c-4840-8fa3-d156fa11086f
# Purpose: Create raw AnnData (.h5ad) without normalization, filtering, or log transform
# ============================================

import scanpy as sc
import pandas as pd
import os

# ------------------------
# Step 1: Load raw AnnData
# ------------------------
print("[Step 1] Loading raw kidney dataset...")

input_path = "./data/raw/kidney/f5efcb4c-99ca-4afb-9f22-af975525e42f.h5ad"
if not os.path.exists(input_path):
    raise FileNotFoundError(f"Cannot find file: {input_path}")

adata = sc.read_h5ad(input_path)
print(f"Loaded AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

# ------------------------
# Step 2: Rename columns for consistency
# ------------------------
print("[Step 2] Renaming metadata columns...")

rename_dict = {
    "donor_id": "patient",
    "cell_type": "cell_type_annotation",
    "disease": "label"
}
adata.obs.rename(columns=rename_dict, inplace=True)
print(f"Columns renamed: {rename_dict}")

# ------------------------
# Step 3: Map disease labels
# ------------------------
print("[Step 3] Mapping disease labels...")

if "label" in adata.obs.columns:
    label_map = {
        "normal": 0,
        "acute kidney failure": 1,
        "chronic kidney disease": 2
    }
    adata.obs["label"] = adata.obs["label"].map(label_map)
    print(f"Label mapping complete: {adata.obs['label'].value_counts(dropna=False).to_dict()}")
else:
    print("'label' column not found. Skipping mapping.")

# ------------------------
# Step 4: Replace var index with feature_name
# ------------------------
print("[Step 4] Setting var index to feature_name...")

if "feature_name" not in adata.var.columns:
    raise KeyError("'feature_name' column not found in adata.var. Please check your input AnnData.")

adata.var["feature_id"] = adata.var.index 
adata.var.index = adata.var["feature_name"].astype(str)

if "feature_name" in adata.var.columns:
    adata.var.drop(columns=["feature_name"], inplace=True)

adata.var.index.name = None

print(f"var.index replaced with feature_name. Example: {adata.var.index[:5].tolist()}")

# ------------------------
# Step 5: Save raw AnnData
# ------------------------
print("[Step 5] Saving raw dataset...")

os.makedirs("./data/kidney", exist_ok=True)
output_path = "./data/kidney/kidney.h5ad"
adata.write_h5ad(output_path, compression="gzip")

print(f"Saved raw AnnData to: {output_path}")
print(f"[INFO] Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"[INFO] Label distribution: {adata.obs['label'].value_counts(dropna=False).to_dict()}")