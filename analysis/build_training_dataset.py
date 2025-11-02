#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility script to build a feature-rich dataset for high-payout modelling."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import train_high_payout_model as base_model

DEFAULT_OUTPUT = Path("analysis") / "model_outputs" / "training_dataset.parquet"


def build_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int,
    output_path: Path,
) -> None:
    results = base_model.load_results(results_path, payout_threshold)
    prerace = base_model.load_prerace(prerace_path)
    entries = base_model.aggregate_entries(entries_path)

    merged = base_model.merge_datasets(results, prerace, entries)
    enriched = base_model.add_derived_features(merged)

    enriched = enriched.sort_values(["race_date", "keirin_cd", "race_no_int"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        enriched.to_parquet(output_path, index=False)
        print(f"saved dataset to {output_path} (rows={len(enriched)})")
    except (ImportError, ValueError):
        fallback_path = output_path.with_suffix(".csv")
        enriched.to_csv(fallback_path, index=False)
        print(
            f"pyarrow/fastparquet missing; saved CSV instead: {fallback_path} (rows={len(enriched)})"
        )

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature dataset for high payout modelling.")
    parser.add_argument(
        "--results",
        default=base_model.DATA_DIR / "keirin_results_20240101_20251004.csv",
        type=Path,
        help="Path to aggregated results CSV.",
    )
    parser.add_argument(
        "--prerace",
        default=base_model.DATA_DIR / "keirin_prerace_20240101_20251004.csv",
        type=Path,
        help="Path to prerace CSV.",
    )
    parser.add_argument(
        "--entries",
        default=base_model.DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv",
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
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"Output path for dataset (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        results_path=args.results,
        prerace_path=args.prerace,
        entries_path=args.entries,
        payout_threshold=args.threshold,
        output_path=args.output,
    )
    print(dataset.head())


if __name__ == "__main__":
    main()
