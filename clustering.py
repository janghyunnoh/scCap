#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering Pipeline for scCap
---------------------------------------
Performs:
  1. HVG filtering
  2. Cluster initialization (Leiden on raw/scGPT space)
  3. Cluster splitting (based on intra/inter distance)
  4. Cluster merging (based on weighted centroid distance)
  5. Cluster selection (evaluate Silhouette / CH / DBI with optional sampling)

Usage Example:
--------------
python clustering.py \
  --input ./data/kidney/kidney_preprocessed.h5ad \
  --output ./data/kidney/kidney_constructed.h5ad \
  --init-space raw \
  --refine-space scgpt \
  --n-hvg 3000 \
  --ratios 2.0 2.1 2.2 \
  --threshold 0.5 \
  --resolution 1.0 \
  --n_neighbors 15 \
  --n_pcs 50 \
  --max-cells 10000
"""

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ============================================================
# Utility
# ============================================================
def get_space_matrix(adata, space="scgpt", n_pcs=50):
    """Return matrix for given space: raw (PCA) or scgpt embedding."""
    if space == "raw":
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, svd_solver="arpack")
        X = adata.obsm["X_pca"]
        X = X[:, :min(n_pcs, X.shape[1])]
    elif space == "scgpt":
        if "X_scGPT" not in adata.obsm:
            raise KeyError("obsm['X_scGPT'] not found")
        X = adata.obsm["X_scGPT"]
    else:
        raise ValueError("space must be 'raw' or 'scgpt'")
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X)


# ============================================================
# Step 0: HVG Filtering
# ============================================================
def filter_hvg(adata, n_hvg=3000):
    """Filter to top n_hvg highly variable genes (if n_hvg > 0)."""
    if n_hvg > 0:
        print(f"[Step 0] Selecting top {n_hvg} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
        print(f" - After HVG filtering: {adata.shape[0]} cells × {adata.shape[1]} genes")
    else:
        print("[Step 0] Skipping HVG filtering (n_hvg=0)")
    return adata


# ============================================================
# Step 1: Initialization
# ============================================================
def cluster_initialization(adata, space="raw", resolution=1.0, n_neighbors=15, n_pcs=50):
    print(f"[Step 1] cluster Initialization (space={space})...")

    if space == "raw":
        adata_tmp = adata.copy()
        sc.tl.pca(adata_tmp, svd_solver="arpack", n_comps=n_pcs)
        sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.leiden(adata_tmp, resolution=resolution)
        adata.obs["init_cluster"] = adata_tmp.obs["leiden"]

    elif space == "scgpt":
        if "X_scGPT" not in adata.obsm:
            raise KeyError("obsm['X_scGPT'] not found in AnnData")
        adata_tmp = adata.copy()
        sc.pp.neighbors(adata_tmp, use_rep="X_scGPT")
        sc.tl.leiden(adata_tmp, resolution=resolution)
        adata.obs["init_cluster"] = adata_tmp.obs["leiden"]

    print(f"[Step 1] Done. {space} clusters={adata.obs['init_cluster'].nunique()}")
    return adata


# ============================================================
# Step 2a: Splitting
# ============================================================
def cluster_splitting(adata, init_cluster, space="scgpt", threshold=0.5):
    print(f"[Step 2 - Splitting] space={space} ...")
    X = get_space_matrix(adata, space)
    unique_labels = np.unique(init_cluster)
    cluster_to_cells = {label: np.where(init_cluster == label)[0] for label in unique_labels}
    centroids = {label: X[idx].mean(axis=0) for label, idx in cluster_to_cells.items()}

    # inter distance
    if len(centroids) >= 2:
        D_cent = pairwise_distances(np.array(list(centroids.values())))
        iu = np.triu_indices(D_cent.shape[0], k=1)
        inter_dist = float(D_cent[iu].mean())
    else:
        inter_dist = 0.0

    new_labels = np.array(init_cluster.astype(int)).copy()
    next_id = int(new_labels.max()) + 1

    for label, cells in cluster_to_cells.items():
        if len(cells) < 2:
            continue
        D_in = pairwise_distances(X[cells])
        iu_in = np.triu_indices(D_in.shape[0], k=1)
        intra_dist = float(D_in[iu_in].mean()) if iu_in[0].size > 0 else 0.0

        if intra_dist > threshold * inter_dist:
            kmedoids = KMedoids(n_clusters=2, method="pam", random_state=0).fit(X[cells])
            for i, idx in enumerate(cells):
                if kmedoids.labels_[i] == 1:
                    new_labels[idx] = next_id
            next_id += 1

    key = f"split_{space}"
    adata.obs[key] = pd.Categorical(new_labels)
    print(f"[Step 2 - Splitting] Result: {len(unique_labels)} → {adata.obs[key].nunique()} clusters")
    return key


# ============================================================
# Step 2b: Merging
# ============================================================
def cluster_merging(adata, cluster_key, space="scgpt", ratio=2.0):
    print(f"[Step 2 - Merging] space={space}, ratio={ratio} ...")
    X = get_space_matrix(adata, space)
    labels = adata.obs[cluster_key].astype(int).values
    merged = labels.copy()

    while True:
        unique = np.unique(merged)
        if len(unique) <= 1:
            break

        clusters = {l: np.where(merged == l)[0] for l in unique}
        centroids = {l: X[idx].mean(axis=0) for l, idx in clusters.items()}
        intra = {l: (np.linalg.norm(X[idx] - centroids[l], axis=1).mean() if len(idx) > 1 else 0.0)
                 for l, idx in clusters.items()}
        avg_intra = np.mean(list(intra.values())) if intra else 0.0

        weighted_dists = {}
        for i, li in enumerate(unique):
            for lj in unique[i+1:]:
                d_inter = np.linalg.norm(centroids[li] - centroids[lj])
                w = avg_intra / (0.5 * (intra[li] + intra[lj] + 1e-8))
                weighted_dists[(li, lj)] = w * d_inter

        if not weighted_dists:
            break
        avg_wdist = np.mean(list(weighted_dists.values()))
        (mi, mj), min_val = min(weighted_dists.items(), key=lambda x: x[1])

        if min_val > avg_wdist / ratio:
            break
        if (np.sum(merged == mi) + np.sum(merged == mj)) > 0.5 * len(merged):
            break
        merged[merged == mj] = mi

    new_labels = pd.Categorical(pd.factorize(merged)[0])
    key = f"merged_{space}_r{ratio}"
    adata.obs[key] = new_labels
    print(f"[Step 2 - Merging] Result: {len(new_labels.categories)} clusters")
    return key


# ============================================================
# Step 3: Selection (with optional sampling)
# ============================================================
def cluster_selection(adata, candidate_keys, space="scgpt",
                    weights=dict(sil=0.4, ch=0.4, db=0.2),
                    max_cells=None, random_state=0):
    """
    Select optimal clustering among candidates based on clustering metrics.
    If max_cells is set and cell count > max_cells, sampling is applied.
    """
    print(f"[Step 3] Selecting optimal cluster (space={space})...")
    np.random.seed(random_state)

    # 샘플링 조건부 적용
    if max_cells is not None and adata.n_obs > max_cells:
        print(f" - Too many cells ({adata.n_obs}), sampling {max_cells} for fast evaluation.")
        sampled_idx = np.random.choice(adata.n_obs, max_cells, replace=False)
        adata_sub = adata[sampled_idx, :].copy()
    else:
        adata_sub = adata

    X = get_space_matrix(adata_sub, space)
    rows = []

    for key in candidate_keys:
        y = adata_sub.obs[key].cat.codes.to_numpy()
        uniq, cnt = np.unique(y, return_counts=True)
        if len(uniq) < 2 or (cnt < 2).any():
            continue
        try:
            sil = silhouette_score(X, y)
        except:
            sil = np.nan
        try:
            ch = calinski_harabasz_score(X, y)
        except:
            ch = np.nan
        try:
            db = davies_bouldin_score(X, y)
        except:
            db = np.nan
        rows.append(dict(key=key, sil=sil, ch=ch, db=db))

    df = pd.DataFrame(rows)
    if df.empty:
        print(" - No valid cluster found for selection.")
        return None, pd.DataFrame()

    # Normalize and weight
    def minmax_norm(values):
        s = pd.Series(values, dtype="float64")
        lo, hi = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-12:
            return pd.Series(np.full(len(s), 0.5), index=s.index)
        return (s - lo) / (hi - lo)

    df["sil_n"] = minmax_norm(df["sil"])
    df["ch_n"] = minmax_norm(df["ch"])
    df["db_inv"] = 1.0 / (1.0 + df["db"])
    df["db_inv_n"] = minmax_norm(df["db_inv"])

    ws, wc, wd = weights["sil"], weights["ch"], weights["db"]
    df["composite"] = ws * df["sil_n"] + wc * df["ch_n"] + wd * df["db_inv_n"]

    best_row = df.sort_values("composite", ascending=False).iloc[0]
    best_key = best_row["key"]

    adata.obs["optimal_cluster"] = adata.obs[best_key]
    print(f"[Step 3] Best={best_key} (score={best_row['composite']:.4f})")
    return "optimal_cluster", df


# ============================================================
# Pipeline
# ============================================================
def run_pipeline(args):
    print(f"[INFO] Loading input file: {args.input}")
    adata = sc.read_h5ad(args.input)

    # Step 0: HVG filtering
    adata = filter_hvg(adata, n_hvg=args.n_hvg)

    # Step 1
    adata = cluster_initialization(adata, space=args.init_space,
                                 resolution=args.resolution,
                                 n_neighbors=args.n_neighbors,
                                 n_pcs=args.n_pcs)

    # Step 2
    candidate_keys = []
    split_key = cluster_splitting(adata, adata.obs["init_cluster"],
                                space=args.refine_space,
                                threshold=args.threshold)
    for r in args.ratios:
        merge_key = cluster_merging(adata, split_key,
                                  space=args.refine_space,
                                  ratio=r)
        candidate_keys.append(merge_key)

    # Step 3
    best_key, df = cluster_selection(adata, candidate_keys,
                                   space=args.refine_space,
                                   max_cells=args.max_cells)

    print(f"[INFO] Saving results to: {args.output}")
    adata.write(args.output)
    print("[INFO] Done.")


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
    parser.add_argument("--ratios", type=float, nargs="+", default=[2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                        help="Merge ratio values (space separated)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Split threshold (intra/inter distance ratio)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Neighbors for graph")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="If set, limits number of cells in selection step for faster evaluation")

    args = parser.parse_args()
    run_pipeline(args)
