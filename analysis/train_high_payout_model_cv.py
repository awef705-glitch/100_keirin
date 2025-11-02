#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a LightGBM model with time-series cross validation for high-payout prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

import train_high_payout_model as base

MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "high_payout_model_lgbm.txt"
METRICS_PATH = MODEL_DIR / "high_payout_model_lgbm_metrics.json"


def build_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:
    results = base.load_results(results_path, payout_threshold)
    prerace = base.load_prerace(prerace_path)
    entries = base.aggregate_entries(entries_path)
    dataset = base.merge_datasets(results, prerace, entries)
    dataset = base.add_derived_features(dataset)
    dataset = dataset.sort_values(["race_date", "keirin_cd", "race_no_int"]).reset_index(drop=True)

    numeric_features, categorical_features = base.select_feature_columns(dataset)

    feature_columns = numeric_features + categorical_features
    missing = [col for col in feature_columns if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = dataset[feature_columns].copy()
    for col in categorical_features:
        X[col] = X[col].astype("category")
    y = dataset["target_high_payout"].astype(int)
    return dataset, X, y, numeric_features, categorical_features


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "class_weight": "balanced",
}


def train_lightgbm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    params: Dict[str, object],
    n_splits: int,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Tuple[Dict[str, object], np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_pred = np.zeros(len(X))
    fold_metrics = []
    best_iterations: List[int] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
        valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features, reference=train_set)

        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=0))

        booster = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        best_iter = booster.best_iteration or num_boost_round
        best_iterations.append(best_iter)

        y_valid_pred = booster.predict(X_valid, num_iteration=best_iter)
        oof_pred[valid_idx] = y_valid_pred

        roc_auc = roc_auc_score(y_valid, y_valid_pred)
        ap = average_precision_score(y_valid, y_valid_pred)
        y_valid_label = (y_valid_pred >= 0.5).astype(int)
        report = classification_report(y_valid, y_valid_label, output_dict=True)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "roc_auc": roc_auc,
                "average_precision": ap,
                "classification_report": report,
                "best_iteration": best_iter,
            }
        )

    overall_roc_auc = roc_auc_score(y, oof_pred)
    overall_ap = average_precision_score(y, oof_pred)
    precision, recall, thresholds = precision_recall_curve(y, oof_pred)
    overall_report = classification_report(y, (oof_pred >= 0.5).astype(int), output_dict=True)

    metrics = {
        "folds": fold_metrics,
        "oof_roc_auc": overall_roc_auc,
        "oof_average_precision": overall_ap,
        "oof_classification_report": overall_report,
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "best_iterations": best_iterations,
    }
    return metrics, oof_pred


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    best_iterations: List[int],
    num_boost_round: int,
    params: Dict[str, object],
) -> lgb.Booster:
    final_rounds = int(np.mean(best_iterations)) if best_iterations else num_boost_round
    final_rounds = max(50, final_rounds)

    full_set = lgb.Dataset(X, label=y, categorical_feature=categorical_features, free_raw_data=False)
    booster = lgb.train(
        params,
        full_set,
        num_boost_round=final_rounds,
        valid_sets=[full_set],
        valid_names=["full"],
    )
    return booster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM high-payout model with CV.")
    parser.add_argument(
        "--results",
        default=base.DATA_DIR / "keirin_results_20240101_20251004.csv",
        type=Path,
        help="Path to aggregated results CSV.",
    )
    parser.add_argument(
        "--prerace",
        default=base.DATA_DIR / "keirin_prerace_20240101_20251004.csv",
        type=Path,
        help="Path to prerace CSV.",
    )
    parser.add_argument(
        "--entries",
        default=base.DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv",
        type=Path,
        help="Path to race entries CSV.",
    )
    parser.add_argument(
        "--threshold",
        default=10000,
        type=int,
        help="Payout threshold (JPY) for positive class.",
    )
    parser.add_argument(
        "--folds",
        default=5,
        type=int,
        help="Number of TimeSeriesSplit folds.",
    )
    parser.add_argument(
        "--num-boost-round",
        default=500,
        type=int,
        help="Maximum number of boosting rounds.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        default=50,
        type=int,
        help="Early stopping rounds for LightGBM.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip saving the final LightGBM model.",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Evaluate a predefined LightGBM parameter grid and pick the best model.",
    )
    parser.add_argument(
        "--top-k",
        default=100,
        type=int,
        help="Top-k races for precision@k calculation (default: 100).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, X, y, _, categorical_features = build_dataset(
        args.results,
        args.prerace,
        args.entries,
        args.threshold,
    )

    base_params = DEFAULT_PARAMS.copy()
    grid_results: List[Dict[str, object]] = []

    if args.grid_search:
        candidate_params = [
            {"learning_rate": 0.05, "num_leaves": 63, "feature_fraction": 0.8, "bagging_fraction": 0.8},
            {"learning_rate": 0.04, "num_leaves": 95, "feature_fraction": 0.85, "bagging_fraction": 0.85},
            {"learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 80, "lambda_l1": 0.0, "lambda_l2": 0.0},
            {"learning_rate": 0.03, "num_leaves": 127, "feature_fraction": 0.75, "bagging_fraction": 0.9},
            {"learning_rate": 0.05, "num_leaves": 63, "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l1": 0.2, "lambda_l2": 0.2},
        ]
        best_result = None

        for idx, param_update in enumerate(candidate_params, start=1):
            params = base_params.copy()
            params.update(param_update)
            metrics_tmp, oof_tmp = train_lightgbm_cv(
                X,
                y,
                categorical_features,
                params=params,
                n_splits=args.folds,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=args.early_stopping_rounds,
            )
            score = metrics_tmp["oof_roc_auc"]
            grid_results.append(
                {
                    "candidate": idx,
                    "params": params,
                    "oof_roc_auc": score,
                    "oof_average_precision": metrics_tmp["oof_average_precision"],
                }
            )
            if not best_result or score > best_result["metrics"]["oof_roc_auc"]:
                best_result = {"params": params, "metrics": metrics_tmp, "oof": oof_tmp}

        assert best_result is not None
        selected_params = best_result["params"]
        metrics = best_result["metrics"]
        oof_pred = best_result["oof"]
    else:
        selected_params = base_params
        metrics, oof_pred = train_lightgbm_cv(
            X,
            y,
            categorical_features,
            params=selected_params,
            n_splits=args.folds,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )

    thresholds_arr = np.array(metrics["thresholds"])
    precision_arr = np.array(metrics["precision_curve"])
    recall_arr = np.array(metrics["recall_curve"])
    if len(thresholds_arr):
        f1_scores = 2 * precision_arr[:-1] * recall_arr[:-1] / (
            precision_arr[:-1] + recall_arr[:-1] + 1e-9
        )
        best_idx = int(np.nanargmax(f1_scores))
        best_threshold = thresholds_arr[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        best_threshold = 0.5
        best_f1 = float("nan")

    order = np.argsort(-oof_pred)
    top_k = max(1, min(args.top_k, len(order)))
    top_indices = order[:top_k]
    precision_at_k = float(y.iloc[top_indices].mean()) if len(top_indices) else float("nan")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    oof_df = dataset[["race_date", "keirin_cd", "race_no_int", "target_high_payout"]].copy()
    oof_df["prediction"] = oof_pred
    oof_df.rename(
        columns={
            "race_no_int": "race_no",
            "target_high_payout": "label",
        },
        inplace=True,
    )
    oof_df.to_csv(MODEL_DIR / "high_payout_model_lgbm_oof.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_save_model:
        booster = train_final_model(
            X,
            y,
            categorical_features,
            metrics.get("best_iterations", []),
            args.num_boost_round,
            params=selected_params,
        )
        booster.save_model(str(MODEL_PATH))
        feature_importance = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance": booster.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        feature_importance.to_csv(
            MODEL_DIR / "high_payout_model_lgbm_feature_importance.csv", index=False
        )

    summary = {
        "payout_threshold": args.threshold,
        "folds": metrics["folds"],
        "oof_roc_auc": metrics["oof_roc_auc"],
        "oof_average_precision": metrics["oof_average_precision"],
        "oof_classification_report": metrics["oof_classification_report"],
        "best_iterations": metrics["best_iterations"],
        "precision_curve": metrics["precision_curve"],
        "recall_curve": metrics["recall_curve"],
        "thresholds": metrics["thresholds"],
        "best_threshold_f1": best_threshold,
        "best_f1_score": best_f1,
        "precision_at_top_k": precision_at_k,
        "top_k": top_k,
        "selected_params": selected_params,
        "grid_search_results": grid_results,
    }
    METRICS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "oof_roc_auc": metrics["oof_roc_auc"],
            "oof_average_precision": metrics["oof_average_precision"],
            "best_iterations": metrics["best_iterations"],
            "best_threshold_f1": best_threshold,
            "best_f1_score": best_f1,
            "precision_at_top_k": precision_at_k,
            "top_k": top_k,
            "selected_params": selected_params,
            "metrics_path": str(METRICS_PATH),
            "model_path": None if args.no_save_model else str(MODEL_PATH),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
