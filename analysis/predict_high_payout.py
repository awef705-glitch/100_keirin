#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate high-payout race predictions using the trained LightGBM model."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_high_payout_model as base_model

MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "high_payout_model_lgbm.txt"
DATASET_PATH = MODEL_DIR / "training_dataset.parquet"
OUTPUT_PATH = MODEL_DIR / "predictions_topk.csv"


def load_dataset(threshold: int) -> pd.DataFrame:
    csv_path = MODEL_DIR / "training_dataset.csv"
    if DATASET_PATH.exists():
        try:
            return pd.read_parquet(DATASET_PATH)
        except (ImportError, ValueError):
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)

    results = base_model.load_results(
        base_model.DATA_DIR / "keirin_results_20240101_20251004.csv",
        threshold,
    )
    prerace = base_model.load_prerace(
        base_model.DATA_DIR / "keirin_prerace_20240101_20251004.csv"
    )
    entries = base_model.aggregate_entries(
        base_model.DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv"
    )
    df = base_model.merge_datasets(results, prerace, entries)
    df = base_model.add_derived_features(df)
    df.to_csv(csv_path, index=False)
    return df


def generate_predictions(
    dataset: pd.DataFrame,
    payout_threshold: int,
    start_date: int | None,
    end_date: int | None,
    top_k: int,
    output_path: Path,
) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    dataset = dataset.sort_values(["race_date", "keirin_cd", "race_no_int"]).reset_index(drop=True)
    if start_date:
        dataset = dataset[dataset["race_date"] >= start_date]
    if end_date:
        dataset = dataset[dataset["race_date"] <= end_date]

    if dataset.empty:
        raise ValueError("No races match the specified date filters.")

    numeric_features, categorical_features = base_model.select_feature_columns(dataset)
    feature_cols = numeric_features + categorical_features

    missing = [col for col in feature_cols if col not in dataset.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    X = dataset[feature_cols].copy()
    for col in categorical_features:
        X[col] = X[col].astype("category")

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    preds = booster.predict(X)

    result_df = dataset[["race_date", "keirin_cd", "race_no_int", "track", "grade"]].copy()
    result_df["prediction"] = preds
    if "target_high_payout" in dataset.columns:
        result_df["label"] = dataset["target_high_payout"]
    result_df = result_df.sort_values("prediction", ascending=False).reset_index(drop=True)

    top_k = max(1, min(top_k, len(result_df)))
    top_predictions = result_df.head(top_k).copy()
    top_predictions = top_predictions.rename(columns={"race_no_int": "race_no"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_predictions.to_csv(output_path, index=False)
    return top_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top high-payout predictions.")
    parser.add_argument("--start-date", type=int, help="Filter races >= YYYYMMDD.")
    parser.add_argument("--end-date", type=int, help="Filter races <= YYYYMMDD.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of races to output (default: 100).")
    parser.add_argument("--threshold", type=int, default=10000, help="High payout threshold (default: 10000).")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.threshold)
    top_predictions = generate_predictions(
        dataset=dataset,
        payout_threshold=args.threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        top_k=args.top_k,
        output_path=args.output,
    )
    print(top_predictions.head(20).to_string(index=False))
    print(f"\nSaved top {len(top_predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
