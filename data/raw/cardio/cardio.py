#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cardiac dataset preprocessing script
-----------------------------------
Downloads from:
https://singlecell.broadinstitute.org/single_cell/study/SCP1303

Required files:
1. DCM_HCM_Expression_Matrix_raw_counts_V1.mtx
2. DCM_HCM_Expression_Matrix_genes_V1.tsv
3. DCM_HCM_Expression_Matrix_barcodes_V1.tsv
4. DCM_HCM_MetaData_V1.txt

Adapted from: https://github.com/Teddy-XiongGZ/ProtoCell4P
"""

from scipy.io import mmread
from scipy import sparse
import scanpy as sc
import pandas as pd
import numpy as np

# ------------------------
# Step 1: Load raw matrix
# ------------------------
print("[Step 1] Loading raw expression matrix...")
data = mmread("./data/raw/cardio/DCM_HCM_Expression_Matrix_raw_counts_V1.mtx")
data = sparse.csr_matrix(data.transpose().astype(np.float32))  # genes × cells → cells × genes
print(f" - Raw matrix shape: {data.shape}")

# ------------------------
# Step 2: Load gene and cell info
# ------------------------
print("[Step 2] Loading gene and barcode info...")
genes = pd.read_csv(
    "./data/raw/cardio/DCM_HCM_Expression_Matrix_genes_V1.tsv",
    sep="\t",
    header=None
).iloc[:, 1].tolist()

barcodes = open("./data/raw/cardio/DCM_HCM_Expression_Matrix_barcodes_V1.tsv") \
    .read().strip().split("\n")

barcodes = [b for b in barcodes]  
print(f" - Genes: {len(genes)}, Cells: {len(barcodes)}")

# ------------------------
# Step 3: Load metadata
# ------------------------
print("[Step 3] Loading metadata...")
meta = pd.read_csv("./data/raw/cardio/DCM_HCM_MetaData_V1.txt", sep="\t", low_memory=False)
meta = meta.drop(index=0).reset_index(drop=True)
meta.set_index("NAME", inplace=True)

# 중복된 index 제거
meta = meta[~meta.index.duplicated(keep="first")]
print(f" - Metadata entries: {meta.shape[0]} (unique)")

# ------------------------
# Step 4: Construct AnnData
# ------------------------
print("[Step 4] Creating AnnData object...")
adata = sc.AnnData(data)
adata.obs.index = barcodes
adata.var.index = genes

# AnnData 중복 인덱스 제거
adata.obs = adata.obs[~adata.obs.index.duplicated(keep="first")]

# 공통 셀만 선택 (index 정합성 확보)
common_idx = adata.obs.index.intersection(meta.index)
adata = adata[common_idx, :].copy()
adata.obs = meta.loc[common_idx, :]

print(f" - Matched unique cells between matrix and metadata: {len(common_idx)}")

# ------------------------
# Step 5: Add labels
# ------------------------
print("[Step 5] Adding disease labels...")
label_map = {
    "normal": 0,
    "hypertrophic cardiomyopathy": 1,
    "dilated cardiomyopathy": 2
}
adata.obs["label"] = adata.obs["disease__ontology_label"].map(label_map)

# Rename obs columns for clarity
adata.obs.rename(columns={
    "donor_id": "patient",
    "cell_type__ontology_label": "cell_type_annotation"
}, inplace=True)

print(f"[INFO] Preprocessing complete. Final AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")

# ------------------------
# Step 6: Save AnnData
# ------------------------
output_file = "./data/cardio/cardio.h5ad"
adata.write_h5ad(output_file)
print(f"[INFO] Saved AnnData to {output_file}")