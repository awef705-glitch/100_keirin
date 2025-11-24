#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis import prerace_model

print("Starting training script...", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM with pre-race features only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/keirin_results_20240101_20251004.csv"),
        help="CSV containing race results (for target creation).",
    )
    parser.add_argument(
        "--prerace",
        type=Path,
        default=Path("data/keirin_prerace_20240101_20251004.csv"),
        help="CSV containing race meta data (schedule information).",
    )
    parser.add_argument(
        "--entries",
        type=Path,
        default=Path("data/keirin_race_detail_entries_20240101_20251004.csv"),
        help="CSV containing rider entries with scores / style / grade.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10000,
        help="Threshold in JPY to define 'high payout' for trifecta.",
    )
    parser.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit folds.")
    parser.add_argument("--num-boost-round", type=int, default=2000, help="Max boosting rounds.")
    parser.add_argument("--early-stopping-rounds", type=int, default=100, help="Early stopping rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Initial learning rate.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-K size for precision@K.")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=prerace_model.MODEL_PATH,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=prerace_model.METADATA_PATH,
        help="Path to save metrics and metadata (JSON).",
    )
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=100,
        help="Print LightGBM metrics every N rounds (0 disables).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization with multiple configurations.",
    )
    return parser.parse_args()


def train_lightgbm(
    dataset: pd.DataFrame,
    feature_columns: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    # Workaround for NumPy 2.0 copy=False issue in pandas astype
    X = dataset[feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    
    y = dataset["target_high_payout"].copy().astype(int)

    tscv = TimeSeriesSplit(n_splits=args.folds)

    # Calculate class weights for imbalanced data
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    scale_pos_weight = float(n_negative / n_positive) if n_positive > 0 else 1.0

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": args.learning_rate,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "scale_pos_weight": scale_pos_weight,  # Address class imbalance
        "verbosity": -1,
        "boost_from_average": True,
        "seed": 42,
    }

    oof_pred = np.zeros(len(dataset), dtype=float)
    feature_gain = pd.Series(0.0, index=feature_columns)
    best_iterations: List[int] = []
    models: List[lgb.Booster] = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        print(f"Starting fold {fold}...")
        try:
            # NumPy 2.0 Fix: Explicitly convert to contiguous arrays
            X_train = np.ascontiguousarray(X.iloc[train_idx].values, dtype=np.float32)
            y_train = np.ascontiguousarray(y.iloc[train_idx].values, dtype=np.int32)
            X_valid = np.ascontiguousarray(X.iloc[valid_idx].values, dtype=np.float32)
            y_valid = np.ascontiguousarray(y.iloc[valid_idx].values, dtype=np.int32)

            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_columns)
            valid_data = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_columns, reference=train_data)
        except Exception as e:
            print(f"Error creating lgb.Dataset: {e}")
            raise e

        callbacks = []
        if args.early_stopping_rounds:
            callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=False))
        if args.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=args.verbose_eval))

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=args.num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        best_iter = booster.best_iteration or args.num_boost_round
        best_iterations.append(best_iter)
        models.append(booster)
        oof_pred[valid_idx] = booster.predict(X.iloc[valid_idx], num_iteration=best_iter)

        gain = pd.Series(booster.feature_importance(importance_type="gain"), index=feature_columns)
        feature_gain += gain

    roc_auc = roc_auc_score(y, oof_pred)
    avg_precision = average_precision_score(y, oof_pred)

    precision, recall, thresholds = precision_recall_curve(y, oof_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = int(np.nanargmax(f1_scores)) if len(f1_scores) else 0
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx]) if len(f1_scores) else 0.0

    order = np.argsort(-oof_pred)
    top_k = min(args.top_k, len(order))
    precision_at_k = float(y.iloc[order[:top_k]].mean())

    report = classification_report(y, (oof_pred >= best_threshold).astype(int), output_dict=True)

    # Retrain on the full dataset with the average best iteration.
    final_iter = max(50, int(np.mean(best_iterations)))
    
    # NumPy 2.0 Fix: Convert full dataset to contiguous arrays
    X_full = np.ascontiguousarray(X.values, dtype=np.float32)
    y_full = np.ascontiguousarray(y.values, dtype=np.int32)
    
    full_data = lgb.Dataset(X_full, label=y_full, feature_name=feature_columns)
    final_callbacks = []
    if args.verbose_eval:
        final_callbacks.append(lgb.log_evaluation(period=args.verbose_eval))

    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_iter,
        valid_sets=[full_data],
        valid_names=["train"],
        callbacks=final_callbacks,
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(args.model_output))

    # Save auxiliary artefacts.
    oof_df = dataset[
        [
            "race_date",
            "keirin_cd",
            "race_no",
            "target_high_payout",
        ]
    ].copy()
    oof_df["prediction"] = oof_pred
    oof_df.rename(columns={"target_high_payout": "label"}, inplace=True)
    prerace_model.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(prerace_model.MODEL_DIR / "prerace_model_oof.csv", index=False)

    feat_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": feature_gain.values / max(1, len(best_iterations)),
        }
    ).sort_values("gain", ascending=False)
    feat_importance.to_csv(
        prerace_model.MODEL_DIR / "prerace_model_feature_importance.csv",
        index=False,
    )

    metrics = {
        "payout_threshold": args.threshold,
        "n_samples": int(len(dataset)),
        "positive_rate": float(y.mean()),
        "folds": args.folds,
        "params": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()},
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision_at_top_k": precision_at_k,
        "top_k": top_k,
        "classification_report": report,
        "best_iterations": best_iterations,
        "final_iteration": final_iter,
    }

    metadata = {
        "feature_columns": feature_columns,
        "best_threshold": best_threshold,
        "high_confidence_threshold": min(0.95, best_threshold + 0.1),
        "payout_threshold": args.threshold,
        "training_samples": int(len(dataset)),
        "positive_rate": float(y.mean()),
        "top_k": top_k,
        "precision_at_top_k": precision_at_k,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "final_iteration": final_iter,
    }

    # Persist artefacts.
    prerace_model.save_training_dataset(dataset)
    prerace_model.save_metadata(metadata)

    metrics_path = args.metrics_output
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return metrics


def optimize_hyperparameters(
    dataset: pd.DataFrame,
    feature_columns: List[str],
    args: argparse.Namespace,
) -> None:
    """Test multiple hyperparameter configurations and report the best."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION MODE")
    print("=" * 70)

    param_configs = [
        {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 7, "min_data_in_leaf": 100},
        {"learning_rate": 0.05, "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 50},
        {"learning_rate": 0.07, "num_leaves": 127, "max_depth": -1, "min_data_in_leaf": 30},
        {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 80},
        {"learning_rate": 0.04, "num_leaves": 50, "max_depth": 8, "min_data_in_leaf": 60},
    ]

    results = []
    for idx, param_config in enumerate(param_configs, start=1):
        print(f"\n[Config {idx}/{len(param_configs)}] Testing: {param_config}")
        temp_args = argparse.Namespace(**vars(args))
        for key, value in param_config.items():
            setattr(temp_args, key, value)

        # Temporarily disable verbose eval for optimization
        temp_args.verbose_eval = 0

        metrics = train_lightgbm(dataset, feature_columns, temp_args)
        results.append({
            "config": param_config,
            "roc_auc": metrics["roc_auc"],
            "avg_precision": metrics["average_precision"],
            "precision_at_k": metrics["precision_at_top_k"],
        })
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | Avg Precision: {metrics['average_precision']:.4f}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    best_result = max(results, key=lambda x: x["roc_auc"])
    print(f"\nBest configuration by ROC-AUC:")
    print(f"  Config: {best_result['config']}")
    print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"  Avg Precision: {best_result['avg_precision']:.4f}")
    print(f"  Precision@Top{args.top_k}: {best_result['precision_at_k']:.4f}")


def main() -> None:
    args = parse_args()
    dataset, feature_columns = prerace_model.build_training_dataset(
        results_path=args.results,
        prerace_path=args.prerace,
        entries_path=args.entries,
        payout_threshold=args.threshold,
    )

    print("=" * 70)
    print("Pre-race dataset summary")
    print("=" * 70)
    print(f"Samples           : {len(dataset):,}")
    print(f"Positive rate     : {dataset['target_high_payout'].mean():.4f}")
    print(f"Feature columns   : {len(feature_columns)}")
    print(f"Output CSV        : {prerace_model.DATASET_PATH}")

    if args.optimize:
        optimize_hyperparameters(dataset, feature_columns, args)
    else:
        metrics = train_lightgbm(dataset, feature_columns, args)

        print("\n" + "=" * 70)
        print("Training finished")
        print("=" * 70)
        print(f"ROC-AUC             : {metrics['roc_auc']:.4f}")
        print(f"Average Precision   : {metrics['average_precision']:.4f}")
        print(f"Best F1 Score       : {metrics['best_f1']:.4f}")
        print(f"Best Threshold      : {metrics['best_threshold']:.4f}")
        print(f"Precision@Top{metrics['top_k']}: {metrics['precision_at_top_k']:.4f}")
        print(f"Model saved to      : {prerace_model.MODEL_PATH}")
        print(f"OOF predictions     : {prerace_model.MODEL_DIR / 'prerace_model_oof.csv'}")
        print(f"Feature importance  : {prerace_model.MODEL_DIR / 'prerace_model_feature_importance.csv'}")
        print(f"Metadata            : {prerace_model.METADATA_PATH}")


if __name__ == "__main__":
    main()
