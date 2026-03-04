#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering Pipeline for scCap (NO SELECTION)
---------------------------------------
Performs:
  1. HVG filtering
  2. Cluster initialization (Leiden on raw/scGPT space)
  3. Cluster splitting (based on intra/inter distance)
  4. Cluster merging (based on weighted centroid distance)

Final output:
  adata.obs["refined_cluster"]

Usage Example:
--------------
python clustering.py \
  --input ./data/kidney/kidney_preprocessed.h5ad \
  --output ./data/kidney/kidney_constructed.h5ad \
  --init-space raw \
  --refine-space scgpt \
  --n-hvg 3000 \
  --ratio 2.0 \
  --resolution 1.0 \
  --threshold 0.5 \
  --constraint 0.5
"""

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances


# ============================================================
# Utility
# ============================================================
def get_space_matrix(adata, space="scgpt", n_pcs=50):
    """Return matrix for given space: raw (PCA) or scGPT embedding."""
    if space == "raw":
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, svd_solver="arpack")
        X = adata.obsm["X_pca"][:, :n_pcs]
    elif space == "scgpt":
        if "X_scGPT" not in adata.obsm:
            raise KeyError("obsm['X_scGPT'] not found")
        X = adata.obsm["X_scGPT"]
    else:
        raise ValueError("space must be 'raw' or 'scgpt'")

    return X.toarray() if sp.issparse(X) else np.asarray(X)


# ============================================================
# Step 0: HVG Filtering
# ============================================================
def filter_hvg(adata, n_hvg=3000):
    if n_hvg > 0:
        print(f"[Step 0] Selecting top {n_hvg} HVGs")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f" - After HVG: {adata.shape}")
    else:
        print("[Step 0] Skip HVG filtering")
    return adata


# ============================================================
# Step 1: Initialization
# ============================================================
def cluster_initialization(
    adata, space="raw", resolution=1.0, n_neighbors=15, n_pcs=50
):
    print(f"[Step 1] Initialization (space={space})")

    adata_tmp = adata.copy()

    if space == "raw":
        sc.tl.pca(adata_tmp, n_comps=n_pcs, svd_solver="arpack")
        sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, n_pcs=n_pcs)
    else:
        sc.pp.neighbors(adata_tmp, use_rep="X_scGPT")

    sc.tl.leiden(adata_tmp, resolution=resolution)
    adata.obs["init_cluster"] = adata_tmp.obs["leiden"]

    print(f" - init clusters: {adata.obs['init_cluster'].nunique()}")
    return adata


# ============================================================
# Step 2a: Splitting
# ============================================================
def cluster_splitting(adata, init_cluster, space="scgpt", threshold=0.5):
    print(f"[Step 2a] Splitting (space={space}, threshold={threshold})")

    X = get_space_matrix(adata, space)
    labels = init_cluster.astype(int).values

    new_labels = labels.copy()
    next_id = labels.max() + 1

    unique = np.unique(labels)
    clusters = {l: np.where(labels == l)[0] for l in unique}
    centroids = {l: X[idx].mean(axis=0) for l, idx in clusters.items()}

    if len(centroids) > 1:
        C = np.vstack(list(centroids.values()))
        inter_dist = pairwise_distances(C).mean()
    else:
        inter_dist = 0.0

    for l, idx in clusters.items():
        if len(idx) < 2:
            continue

        intra = pairwise_distances(X[idx]).mean()
        if intra > threshold * inter_dist:
            km = KMedoids(n_clusters=2, random_state=0).fit(X[idx])
            for i, cell in enumerate(idx):
                if km.labels_[i] == 1:
                    new_labels[cell] = next_id
            next_id += 1

    adata.obs["split_cluster"] = pd.Categorical(new_labels)
    print(f" - after split: {adata.obs['split_cluster'].nunique()} clusters")
    return "split_cluster"


# ============================================================
# Step 2b: Merging
# ============================================================
def cluster_merging(
    adata, cluster_key, space="scgpt", ratio=2.0, constraint=0.5
):
    print(f"[Step 2b] Merging (space={space}, ratio={ratio}, constraint={constraint})")

    X = get_space_matrix(adata, space)
    labels = adata.obs[cluster_key].astype(int).values
    merged = labels.copy()

    while True:
        uniq = np.unique(merged)
        if len(uniq) <= 1:
            break

        clusters = {l: np.where(merged == l)[0] for l in uniq}
        centroids = {l: X[idx].mean(axis=0) for l, idx in clusters.items()}
        intra = {
            l: np.linalg.norm(X[idx] - centroids[l], axis=1).mean()
            if len(idx) > 1 else 0.0
            for l, idx in clusters.items()
        }

        avg_intra = np.mean(list(intra.values()))
        dists = {}

        for i, li in enumerate(uniq):
            for lj in uniq[i + 1:]:
                d = np.linalg.norm(centroids[li] - centroids[lj])
                w = avg_intra / (0.5 * (intra[li] + intra[lj] + 1e-8))
                dists[(li, lj)] = w * d

        (mi, mj), v = min(dists.items(), key=lambda x: x[1])
        if v > np.mean(list(dists.values())) / ratio:
            break
        if (merged == mi).sum() + (merged == mj).sum() > constraint * len(merged):
            break

        merged[merged == mj] = mi

    adata.obs["refined_cluster"] = pd.Categorical(pd.factorize(merged)[0])
    print(f" - final clusters: {adata.obs['refined_cluster'].nunique()}")
    return adata


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(args):
    print(f"[INFO] Loading {args.input}")
    adata = sc.read_h5ad(args.input)

    adata = filter_hvg(adata, args.n_hvg)

    adata = cluster_initialization(
        adata,
        space=args.init_space,
        resolution=args.resolution,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
    )

    split_key = cluster_splitting(
        adata,
        adata.obs["init_cluster"],
        space=args.refine_space,
        threshold=args.threshold,
    )

    adata = cluster_merging(
        adata,
        split_key,
        space=args.refine_space,
        ratio=args.ratio,
        constraint=args.constraint,
    )

    print(f"[INFO] Saving to {args.output}")
    adata.write(args.output)
    print("[INFO] Done")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster construction with split/merge and selection")

    parser.add_argument("--input", type=str, required=True, help="Input .h5ad file")
    parser.add_argument("--output", type=str, required=True, help="Output .h5ad file")
    parser.add_argument("--init-space", type=str, choices=["raw", "scgpt"], default="raw",
                        help="Representation space for cluster initialization")
    parser.add_argument("--refine-space", type=str, choices=["raw", "scgpt"], default="scgpt",
                        help="Representation space for refinement (split/merge/selection)")
    parser.add_argument("--n-hvg", type=int, default=3000,
                        help="Number of highly variable genes to select (0 to skip)")
    parser.add_argument("--ratio", type=float, default=2.0,
                        help="Merge ratio values (space separated)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Split threshold (intra/inter distance ratio)")
    parser.add_argument("--constraint", type=float, default=0.5,
                        help="Max cluster size ratio constraint (prevent merging if result exceeds this fraction of total cells)")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Neighbors for graph")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="If set, limits number of cells in selection step for faster evaluation")

    args = parser.parse_args()
    run_pipeline(args)
