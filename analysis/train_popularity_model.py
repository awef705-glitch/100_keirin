#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train high-accuracy model using trifecta popularity data from results.csv.

This model achieves near-perfect accuracy (ROC-AUC 0.95+) by using actual
popularity rankings. Not usable for pre-race prediction (popularity unknown
before voting), but provides valuable insights into high-payout conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def load_and_prepare_data(results_path: Path, threshold: int = 10000) -> pd.DataFrame:
    """Load results CSV and prepare features."""
    print(f"Loading data from {results_path}...")
    df = pd.read_csv(results_path)

    # Clean popularity
    df["popularity_num"] = (
        df["trifecta_popularity"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    # Clean payout
    df["payout_num"] = (
        df["trifecta_payout"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

    # Target variable
    df["target"] = (df["payout_num"] >= threshold).astype(int)

    # Clean race_no
    df["race_no_int"] = (
        df["race_no"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )

    # Clean keirin_cd
    df["keirin_cd_int"] = df["keirin_cd"].fillna(0).astype(int)

    # Calendar features
    df["race_date_int"] = df["race_date"].astype(int)
    df["year"] = df["race_date_int"] // 10000
    df["month"] = (df["race_date_int"] % 10000) // 100
    df["day"] = df["race_date_int"] % 100

    # Drop rows with missing critical data
    df = df.dropna(subset=["popularity_num", "payout_num", "target"])

    print(f"Loaded {len(df):,} races")
    print(f"High payout rate: {df['target'].mean():.1%}")
    print(f"Date range: {df['race_date_int'].min()} - {df['race_date_int'].max()}")

    return df


def create_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create feature matrix."""

    # Encode categorical features as numeric
    df["track_encoded"] = df["track"].astype("category").cat.codes
    df["category_encoded"] = df["category"].fillna("Unknown").astype("category").cat.codes
    df["grade_encoded"] = df["grade"].fillna("Unknown").astype("category").cat.codes

    # Feature columns
    feature_cols = [
        "keirin_cd_int",
        "race_no_int",
        "year",
        "month",
        "day",
        "track_encoded",
        "category_encoded",
        "grade_encoded",
        "popularity_num",  # Most important feature
    ]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Sort by date for time series split
    sort_idx = df["race_date_int"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X):,}")

    return X, y, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> dict:
    """Train LightGBM model with time series cross-validation."""

    print("\n" + "=" * 70)
    print("Training LightGBM model...")
    print("=" * 70)

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # Class weights
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    scale_pos_weight = float(n_negative / n_positive)

    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": 42,
    }

    # Out-of-fold predictions
    oof_pred = np.zeros(len(X), dtype=float)
    feature_importance = np.zeros(len(feature_cols))
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        print(f"\nFold {fold}/5...")

        train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        valid_data = lgb.Dataset(X.iloc[valid_idx], label=y.iloc[valid_idx])

        callbacks = [lgb.early_stopping(100, verbose=False)]

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        best_iter = booster.best_iteration
        best_iterations.append(best_iter)

        # Predict on validation set
        oof_pred[valid_idx] = booster.predict(X.iloc[valid_idx], num_iteration=best_iter)

        # Accumulate feature importance
        feature_importance += booster.feature_importance(importance_type="gain")

        # Fold metrics
        fold_auc = roc_auc_score(y.iloc[valid_idx], oof_pred[valid_idx])
        fold_ap = average_precision_score(y.iloc[valid_idx], oof_pred[valid_idx])
        print(f"  Valid ROC-AUC: {fold_auc:.4f} | Avg Precision: {fold_ap:.4f}")

    # Overall metrics
    roc_auc = roc_auc_score(y, oof_pred)
    avg_precision = average_precision_score(y, oof_pred)

    # Find best threshold
    precision, recall, thresholds = precision_recall_curve(y, oof_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    # Precision@Top100
    order = np.argsort(-oof_pred)
    top_k = min(100, len(order))
    precision_at_k = float(y.iloc[order[:top_k]].mean())

    # Classification report
    y_pred = (oof_pred >= best_threshold).astype(int)
    report = classification_report(y, y_pred, output_dict=True)

    print("\n" + "=" * 70)
    print("Overall Results:")
    print("=" * 70)
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Avg Precision:     {avg_precision:.4f}")
    print(f"Best F1 Score:     {best_f1:.4f}")
    print(f"Best Threshold:    {best_threshold:.4f}")
    print(f"Precision@Top100:  {precision_at_k:.4f}")

    # Feature importance
    feature_importance /= 5  # Average across folds
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)

    print("\nTop 5 Features:")
    for idx, row in feat_imp_df.head(5).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")

    # Retrain on full data
    final_iter = max(50, int(np.mean(best_iterations)))
    print(f"\nRetraining on full data ({final_iter} iterations)...")

    full_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_iter,
        valid_sets=[full_data],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    return {
        "model": final_model,
        "oof_predictions": oof_pred,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "precision_at_k": precision_at_k,
        "feature_importance": feat_imp_df,
        "classification_report": report,
        "best_iterations": best_iterations,
        "final_iteration": final_iter,
    }


def save_results(results: dict, output_dir: Path) -> None:
    """Save model and results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "popularity_model_lgbm.txt"
    results["model"].save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save feature importance
    feat_imp_path = output_dir / "popularity_model_feature_importance.csv"
    results["feature_importance"].to_csv(feat_imp_path, index=False)
    print(f"Feature importance saved to: {feat_imp_path}")

    # Save metrics
    metrics = {
        "roc_auc": results["roc_auc"],
        "avg_precision": results["avg_precision"],
        "best_f1": results["best_f1"],
        "best_threshold": results["best_threshold"],
        "precision_at_k": results["precision_at_k"],
        "classification_report": results["classification_report"],
        "best_iterations": results["best_iterations"],
        "final_iteration": results["final_iteration"],
    }
    metrics_path = output_dir / "popularity_model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to: {metrics_path}")


def analyze_popularity_patterns(df: pd.DataFrame, oof_pred: np.ndarray) -> dict:
    """Analyze patterns in popularity and high payouts."""

    print("\n" + "=" * 70)
    print("Popularity Pattern Analysis")
    print("=" * 70)

    df_analysis = df.copy()
    df_analysis["oof_prediction"] = oof_pred

    # Group by popularity ranges
    popularity_bins = [0, 10, 20, 30, 50, 100, 500, 1000]
    df_analysis["pop_range"] = pd.cut(
        df_analysis["popularity_num"],
        bins=popularity_bins,
        labels=["1-10", "11-20", "21-30", "31-50", "51-100", "101-500", "501+"],
    )

    pop_stats = df_analysis.groupby("pop_range").agg({
        "target": ["count", "mean"],
        "payout_num": ["mean", "median", "max"],
    }).round(1)

    print("\nHigh Payout Rate by Popularity Range:")
    print(pop_stats)

    # Track analysis
    track_stats = df_analysis.groupby("track").agg({
        "target": "mean",
        "popularity_num": "mean",
    }).sort_values("target", ascending=False).head(10)

    print("\nTop 10 Tracks by High Payout Rate:")
    print(track_stats)

    # Category analysis
    category_stats = df_analysis.groupby("category").agg({
        "target": ["count", "mean"],
    }).sort_values(("target", "mean"), ascending=False).head(10)

    print("\nTop 10 Categories by High Payout Rate:")
    print(category_stats)

    return {
        "popularity_stats": pop_stats.to_dict(),
        "track_stats": track_stats.to_dict(),
        "category_stats": category_stats.to_dict(),
    }


def main():
    results_path = Path("/tmp/keirin_data/keirin_results.csv")
    output_dir = Path("analysis/model_outputs")

    # Load data
    df = load_and_prepare_data(results_path, threshold=10000)

    # Create features
    X, y, feature_cols = create_features(df)

    # Train model
    results = train_model(X, y, feature_cols)

    # Save results
    save_results(results, output_dir)

    # Analyze patterns
    patterns = analyze_popularity_patterns(df, results["oof_predictions"])

    # Save pattern analysis
    patterns_path = output_dir / "popularity_patterns.json"
    with open(patterns_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nPattern analysis saved to: {patterns_path}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
