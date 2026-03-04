import argparse
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import scanpy as sc
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train import (_compute_auprc, _compute_roc_auc, _get_prediction_pairs,
                   objective_wrapper, repeated_k_fold, train_and_tune,
                   predict_and_save)
from run_permute import get_p_val_cell_type
from utils import get_df, get_meta, get_data, train, predict
from vary_data_quality import vary_cell_count, vary_train_size, randomize_cell_annot


def load_presplit_pairs(split_root: str):
    base = Path(split_root)
    if not base.exists():
        raise FileNotFoundError(f"split root not found: {split_root}")
    pairs = []
    for repeat_dir in sorted(base.glob("repeat_*"), key=lambda p: int(p.name.split("_")[1])):
        repeat_idx = int(repeat_dir.name.split("_")[1])
        for train_path in sorted(repeat_dir.glob("fold_*_train.h5ad"), key=lambda p: int(p.name.split("_")[1])):
            fold_idx = int(train_path.name.split("_")[1])
            test_path = repeat_dir / f"fold_{fold_idx}_test.h5ad"
            if not test_path.exists():
                continue
            pairs.append((repeat_idx, fold_idx, train_path, test_path))
    return pairs


def run_task_7_presplit(args, meta_cols):
    if args.split_root is None:
        raise ValueError("--split-root must be provided for task 7")
    splits = load_presplit_pairs(args.split_root)
    if not splits:
        raise ValueError(f"No splits found under {args.split_root}")
    rows = []
    metric_cols = ["auc", "auprc", "accuracy", "precision", "recall", "f1"]
    for repeat_idx, fold_idx, train_path, test_path in splits:
        train_adata = sc.read_h5ad(train_path)
        test_adata = sc.read_h5ad(test_path)
        combined = sc.concat([train_adata, test_adata], join="inner", label="split", keys=["train", "test"])

        args.all_ct = combined.obs[args.cell_type_annot_key].unique()
        args.n_classes = len(set(combined.obs[args.label_key]))
        args.binary = args.n_classes == 2

        df = get_df(combined, args.patient_id_key, args.label_key, args.cell_type_annot_key)
        meta = get_meta(combined, meta_cols, args.patient_id_key) if args.use_meta else None

        train_samples = train_adata.obs[[args.patient_id_key, args.label_key]].drop_duplicates().reset_index(drop=True)
        test_samples = test_adata.obs[[args.patient_id_key, args.label_key]].drop_duplicates().reset_index(drop=True)

        sampler = TPESampler(seed=0)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective_wrapper(df, meta, train_samples, args), n_trials=args.n_tune_trials)
        best_params = study.best_params

        X_train, y_train, batch_train, meta_train = get_data(df, args.all_ct, train_samples,
                                                            binary=args.binary,
                                                            meta=meta if args.use_meta else None,
                                                            attn2=args.attn2)
        X_test, y_test, batch_test, meta_test = get_data(df, args.all_ct, test_samples,
                                                         binary=args.binary,
                                                         meta=meta if args.use_meta else None,
                                                         attn2=args.attn2)

        model = train(X_train, y_train, batch_train, meta_train, args,
                      dropout=best_params["dropout"],
                      n_layers_lin=best_params["n_layers_lin"],
                      n_layers_lin_meta=1 if not args.use_meta else best_params["n_layers_lin_meta"],
                      n_hid=best_params["n_hid"],
                      lr=best_params["lr"],
                      weight_decay=best_params["weight_decay"],
                      n_epochs=best_params["n_epochs"],
                      seed=repeat_idx)

        pred = predict(model, X_test, batch_test, meta_test, len(test_samples), args).cpu().numpy()
        truth = y_test.cpu().numpy()

        y_score, y_pred = _get_prediction_pairs(pred, args)
        y_true = truth.astype(int)
        auc = _compute_roc_auc(y_true, y_score, args)
        aprc = _compute_auprc(y_true, y_score, args)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        rows.append({
            "repeat": repeat_idx,
            "fold": fold_idx,
            "auc": float(auc),
            "auprc": float(aprc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    res_df = pd.DataFrame(rows)
    repeat_summary = res_df.groupby("repeat")[metric_cols].mean().reset_index()
    repeat_summary.to_csv(args.output, index=False)
    metrics = {
        name: (
            float(repeat_summary[name].mean()),
            float(np.std(repeat_summary[name].values, ddof=0)),
        )
        for name in metric_cols
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--meta_cols", default=None)
    parser.add_argument("--output", default="out.txt")
    parser.add_argument("--model_save_path", default="model.pt")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--patient_id_key", default="patient")
    parser.add_argument("--label_key", default="label")
    parser.add_argument("--cell_type_annot_key", default="cell_type_annotation")
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--n_folds_hyperparam_tune", type=int, default=10)
    parser.add_argument("--n_perm", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn1", type=int, default=1)
    parser.add_argument("--attn2", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--n_layers_lin", type=int, default=1)
    parser.add_argument("--n_layers_lin_meta", type=int, default=1)
    parser.add_argument("--n_hid", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_tune_trials", type=int, default=30)

    args = parser.parse_args()

    args.use_meta = True if args.meta_cols is not None else False
    meta_cols = []
    if args.use_meta:
        with open(args.meta_cols, "r") as file:
            meta_cols = [line.strip() for line in file if line.strip()]

    df = None
    meta = None
    if args.task == 7:
        derived_split_root = args.split_root or args.data_path
        if derived_split_root is None:
            raise ValueError("--split-root or --data_path must be provided for task 7")
        args.split_root = derived_split_root
    else:
        adata = sc.read_h5ad(args.data_path)
        args.n_classes = len(set(adata.obs[args.label_key]))
        args.binary = args.n_classes == 2
        args.all_ct = adata.obs[args.cell_type_annot_key].unique()
        df = get_df(adata, args.patient_id_key, args.label_key, args.cell_type_annot_key, no_label=(args.task == 1))
        if args.use_meta:
            meta = get_meta(adata, meta_cols, args.patient_id_key)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDNN_DETERMINISTIC"] = "1"
    start_time = time.time()

    if args.task == 0:
        model = train_and_tune(df, meta, args)
        print(f"Model saved to {args.model_save_path}")

    if args.task == 1:
        pred = predict_and_save(df, meta, args)
        print(f"Predictions saved to {args.output}")

    if args.task == 2:
        metrics = repeated_k_fold(df, meta, args)
        for key, label in [
            ("auc", "AUC"),
            ("auprc", "AUPRC"),
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]:
            mean, std = metrics[key]
            print(f"{label}: {mean:.2f}±{std:.2f}")
        print(f"Results saved to {args.output}")

    if args.task == 3:
        train_sizes = [0.25, 0.5, 0.75]
        res = vary_train_size(train_sizes, df, args)
        print(f"Results saved to {args.output}")

    if args.task == 4:
        cell_counts = [0.25, 0.5, 0.75]
        res = vary_cell_count(cell_counts, df, args)
        print(f"Results saved to {args.output}")

    if args.task == 5:
        cell_props = [0.25, 0.5]
        res = randomize_cell_annot(cell_props, df, args)
        print(f"Results saved to {args.output}")

    if args.task == 6:
        res = get_p_val_cell_type(df, args)
        print(f"Results saved to {args.output}")

    if args.task == 7:
        metrics = run_task_7_presplit(args, meta_cols)
        for key, label in [
            ("auc", "AUC"),
            ("auprc", "AUPRC"),
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
        ]:
            mean, std = metrics[key]
            print(f"{label}: {mean:.2f}±{std:.2f}")
        print(f"Results saved to {args.output}")

    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Total execution time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")


if __name__ == "__main__":
    main()
