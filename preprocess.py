#!/usr/bin/env python3
"""
Generic preprocessing pipeline for scCap datasets.
Supports COVID, Cardio, Kidney datasets or any compatible single-cell dataset.
"""

import argparse
import scanpy as sc
import scgpt as scg
import pandas as pd
from pathlib import Path
import warnings


# ------------------------
# Step 1: Preprocessing
# ------------------------
def preprocess_for_scgpt(adata, min_cells=5, target_sum=1e4):
    """
    Preprocess AnnData for scGPT embedding.
    - Filter genes detected in less than `min_cells`
    - Normalize counts per cell to `target_sum`
    - Log1p transform
    """
    print("[Step 1] Preprocessing data for scGPT...")
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    print(f" - After preprocessing: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata


# ------------------------
# Step 2: scGPT Embedding
# ------------------------
def add_scgpt_embedding(adata, model_dir, batch_size=128):
    """
    Generate scGPT embeddings and store in adata.obsm['X_scGPT'].
    """
    print("[Step 2] Generating scGPT embedding...")

    # Ensure Gene column for scGPT input
    adata.var["Gene"] = adata.var.index

    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir=Path(model_dir),
        gene_col="Gene",
        batch_size=batch_size,
        device="cuda:0",  # Always use GPU by default
        return_new_adata=True,
    )

    embed_adata.obs = adata.obs
    embed_adata.raw = adata.copy()

    # Add embeddings to the original AnnData
    adata.obsm["X_scGPT"] = embed_adata.X
    print(f" - Embedding added to adata.obsm['X_scGPT'] with shape {adata.obsm['X_scGPT'].shape}")
    return adata


# ------------------------
# Step 3: Add manual + SingleR annotations (optional)
# ------------------------
def add_annotations(adata, singler_csv):
    """
    - Rename `cell_type_annotation` → `manual_annotation` (if exists)
    - Add SingleR annotations from CSV file (optional)
    """
    print("[Step 3] Adding manual and SingleR annotations...")

    if "cell_type_annotation" in adata.obs:
        adata.obs["manual_annotation"] = adata.obs["cell_type_annotation"]
        adata.obs.drop(columns=["cell_type_annotation"], inplace=True)
        print(" - Renamed 'cell_type_annotation' → 'manual_annotation'")

    if singler_csv is not None:
        singler_df = pd.read_csv(singler_csv)
        singler_df.set_index("Cell", inplace=True)
        common_cells = adata.obs.index.intersection(singler_df.index)
        adata.obs.loc[common_cells, "singler_annotation"] = singler_df.loc[common_cells, "SingleR_Label"]
        print(f" - Added SingleR annotations for {len(common_cells)} cells")
    else:
        print(" - Skipping SingleR augmentation (no CSV provided).")

    return adata


# ------------------------
# Main pipeline
# ------------------------
def main(args):
    print(f"[INFO] Loading input file: {args.input}")
    adata = sc.read_h5ad(args.input)

    # Step 1: Preprocessing
    adata = preprocess_for_scgpt(adata, min_cells=args.min_cells, target_sum=args.target_sum)

    # Step 2: scGPT embedding
    adata = add_scgpt_embedding(adata, args.model, batch_size=args.batch_size)

    # Step 3: Add manual and (optional) SingleR annotations
    adata = add_annotations(adata, args.singler)

    # Save final result
    print(f"[INFO] Saving final AnnData with scGPT embedding + annotations to: {args.output}")
    adata.write(args.output)
    print("[INFO] Done.")


# ------------------------
# CLI Interface
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic preprocessing pipeline for scCap datasets")

    parser.add_argument("--input", type=str, required=True, help="Input .h5ad file path")
    parser.add_argument("--output", type=str, required=True, help="Output .h5ad file path")
    parser.add_argument("--singler", type=str, default=None, help="Path to SingleR annotation CSV (optional)")

    # Default model directory now set
    parser.add_argument("--model", type=str, default="./scGPT/model_human", help="Path to pretrained scGPT model directory (default: ./scGPT/model_human)")
    parser.add_argument("--min_cells", type=int, default=5, help="Minimum number of cells for gene filtering")
    parser.add_argument("--target_sum", type=float, default=1e4, help="Target sum for normalization")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for scGPT embedding")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ResourceWarning)
    warnings.filterwarnings("ignore", message="flash-attn is not installed, using pytorch transformer instead.")
    warnings.filterwarnings("ignore", category=FutureWarning)

    main(args)
