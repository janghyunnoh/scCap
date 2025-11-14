# ============================================
# COVID Nasal Swab scRNA-seq preprocessing
# From: https://singlecell.broadinstitute.org/single_cell/study/SCP1289/
# Files required:
#  1. 20210220_NasalSwab_RawCounts.txt
#  2. 20210701_NasalSwab_MetaData.txt
#  (Optional) 20210220_NasalSwab_NormCounts.txt
# ============================================

import pandas as pd
import scanpy as sc
import os

# ------------------------
# Step 1: Load raw counts
# ------------------------
print("[Step 1] Loading raw counts matrix...")
raw_path = "./data/raw/covid/20210220_NasalSwab_RawCounts.txt"

if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Cannot find {raw_path}. Please download the file first.")

df = pd.read_csv(raw_path, sep="\t")
print(f" - Loaded matrix: {df.shape[0]} genes × {df.shape[1]-1} cells")

# Transpose to cells × genes (scanpy format)
adata = sc.AnnData(df.T.astype("float32"))
adata.obs.index = df.columns
adata.var.index = df.index

print(f" - AnnData shape: {adata.shape}")

# ------------------------
# Step 2: Load metadata
# ------------------------
print("[Step 2] Loading metadata...")
meta_path = "./data/raw/covid/20210701_NasalSwab_MetaData.txt"

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Cannot find {meta_path}. Please download the file first.")

meta = pd.read_csv(meta_path, sep="\t").drop(axis=0, index=0).reset_index(drop=True)
meta.set_index("NAME", inplace=True)
print(f" - Metadata entries: {meta.shape[0]}")

# Align metadata with AnnData.obs (intersection only)
common_idx = adata.obs.index.intersection(meta.index)
adata = adata[common_idx, :].copy()
adata.obs = meta.loc[common_idx, :]
print(f" - Matched {len(common_idx)} cells with metadata")

# ------------------------
# Step 3: Add labels
# ------------------------
print("[Step 3] Adding disease labels...")
adata.obs["label"] = adata.obs["disease__ontology_label"].apply(
    lambda x: 0 if x == "normal" else 1 if x == "COVID-19" else -1
)
adata = adata[adata.obs["label"] != -1]

# ------------------------
# Step 4: Add cell type & patient info + rename columns
# ------------------------
print("[Step 4] Adding cell type annotations and patient IDs...")

# Add safely (only if not already present)
if "cell_type_annotation" not in adata.obs.columns:
    adata.obs["cell_type_annotation"] = adata.obs.get("Coarse_Cell_Annotations", "unknown")
if "patient" not in adata.obs.columns:
    adata.obs["patient"] = adata.obs.get("donor_id", "unknown")

# Rename for clarity (remove duplicates first)
rename_dict = {
    "donor_id": "patient",
    "Coarse_Cell_Annotations": "original_cell_type_annotation"
}
for old, new in rename_dict.items():
    if old in adata.obs.columns:
        # 만약 동일 이름이 이미 있으면 먼저 삭제
        if new in adata.obs.columns:
            del adata.obs[new]
        adata.obs.rename(columns={old: new}, inplace=True)

print(" - Renamed columns for consistency")

# ------------------------
# Step 5: Save AnnData
# ------------------------
os.makedirs("./data/covid", exist_ok=True)
output_file = "./data/covid/covid.h5ad"
adata.write_h5ad(output_file)

print(f"[INFO] Saved AnnData to {output_file}")
print(f"[INFO] Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"[INFO] Disease labels: {adata.obs['label'].value_counts().to_dict()}")