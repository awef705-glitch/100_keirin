#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a baseline classifier that predicts high-payout trifecta races."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_DIR = Path("data")
MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "high_payout_model.joblib"
METRICS_PATH = MODEL_DIR / "high_payout_metrics.json"

STYLE_ALIAS = {
    "両": "style_ryo",
    "追": "style_oikomi",
    "逃": "style_nige",
}


def sanitize_style(name: str) -> str:
    if name in STYLE_ALIAS:
        return STYLE_ALIAS[name]
    hex_code = "".join(f"{ord(char):x}" for char in name)
    return f"style_{hex_code}"


def load_results(path: Path, payout_threshold: int) -> pd.DataFrame:
    usecols = [
        "race_date",
        "keirin_cd",
        "race_no",
        "track",
        "grade",
        "category",
        "meeting_icon",
        "trifecta_payout",
        "trifecta_popularity",
    ]
    df = pd.read_csv(path, usecols=usecols)
    df["trifecta_payout"] = (
        df["trifecta_payout"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    df["trifecta_popularity"] = (
        df["trifecta_popularity"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    df = df.dropna(subset=["trifecta_payout"])
    df["target_high_payout"] = (df["trifecta_payout"] >= payout_threshold).astype(int)
    df["race_no_int"] = (
        df["race_no"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype(pd.Int64Dtype())
    )
    df = df.dropna(subset=["race_no_int"])
    df["race_no_int"] = df["race_no_int"].astype(int)
    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)
    df["race_date"] = df["race_date"].astype(int)
    return df


def load_prerace(path: Path) -> pd.DataFrame:
    usecols = [
        "race_date",
        "bkeirin_cd",
        "race_no",
        "race_encp",
        "entry_count",
        "narabi_flg",
        "narabi_y_cnt",
        "seri",
        "ozz_flg",
        "vote_flg",
        "grade",
    ]
    df = pd.read_csv(path, usecols=usecols, dtype={"bkeirin_cd": str})
    df["race_date"] = df["race_date"].astype(int)
    df["race_no"] = df["race_no"].astype(int)
    return df


def aggregate_entries(path: Path) -> pd.DataFrame:
    usecols = [
        "race_encp",
        "heikinTokuten",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
        "kyakusitu",
        "syaban",
    ]
    df = pd.read_csv(path, usecols=usecols)
    numeric_cols = [
        "heikinTokuten",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    agg_numeric = (
        df.groupby("race_encp")[numeric_cols]
        .agg(["mean", "std", "min", "max"])
    )
    agg_numeric.columns = [f"{col}_{stat}" for col, stat in agg_numeric.columns]
    style_counts = (
        df.pivot_table(
            index="race_encp",
            columns="kyakusitu",
            values="syaban",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns=lambda name: sanitize_style(name if isinstance(name, str) else ""))
    )
    return agg_numeric.join(style_counts, how="left").reset_index()


def merge_datasets(
    results: pd.DataFrame,
    prerace: pd.DataFrame,
    entries_agg: pd.DataFrame,
) -> pd.DataFrame:
    prerace_enriched = prerace.merge(entries_agg, on="race_encp", how="left")
    merged = results.merge(
        prerace_enriched,
        left_on=["race_date", "keirin_cd", "race_no_int"],
        right_on=["race_date", "bkeirin_cd", "race_no"],
        how="left",
        suffixes=("", "_pre"),
    )
    merged = merged.drop(columns=["bkeirin_cd", "race_no"])
    return merged


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denom = denominator.replace({0: np.nan})
        return numerator / denom

    result = df.copy()

    result["heikinTokuten_range"] = (
        result["heikinTokuten_max"] - result["heikinTokuten_min"]
    )
    result["heikinTokuten_cv"] = safe_div(
        result["heikinTokuten_std"], result["heikinTokuten_mean"].abs()
    )

    for prefix in ["nigeCnt", "makuriCnt", "sasiCnt", "markCnt", "backCnt"]:
        if f"{prefix}_max" in result.columns:
            result[f"{prefix}_range"] = (
                result[f"{prefix}_max"] - result[f"{prefix}_min"]
            )
            result[f"{prefix}_cv"] = safe_div(
                result[f"{prefix}_std"], result[f"{prefix}_mean"].abs()
            )

    style_cols = [col for col in result.columns if col.startswith("style_")]
    if style_cols:
        result["style_total"] = result[style_cols].sum(axis=1)
        for col in style_cols:
            result[f"{col}_ratio"] = safe_div(result[col], result["style_total"])
    else:
        result["style_total"] = np.nan

    if "entry_count" in result.columns:
        result["entry_intensity"] = safe_div(
            result["entry_count"],
            result["style_total"],
        )
    if "narabi_y_cnt" in result.columns and "entry_count" in result.columns:
        result["narabi_ratio"] = safe_div(result["narabi_y_cnt"], result["entry_count"])

    if "trifecta_popularity" in result.columns:
        missing_mask = result["trifecta_popularity"].isna()
        result["popularity_missing"] = missing_mask.astype(int)
        median_value = result["trifecta_popularity"].median()
        result["trifecta_popularity"] = result["trifecta_popularity"].fillna(median_value)

    if "race_no_int" in result.columns and "entry_count" in result.columns:
        result["race_no_norm"] = safe_div(
            result["race_no_int"], result["entry_count"]
        )

    return result.replace([np.inf, -np.inf], np.nan)


def select_feature_columns(dataset: pd.DataFrame) -> Tuple[List[str], List[str]]:
    base_numeric = [
        "trifecta_popularity",
        "entry_count",
        "narabi_flg",
        "narabi_y_cnt",
        "seri",
        "ozz_flg",
        "vote_flg",
        "entry_intensity",
        "narabi_ratio",
        "race_no_norm",
        "popularity_missing",
    ]

    stat_prefixes = [
        "heikinTokuten",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
    ]
    stats = ["mean", "std", "min", "max", "range", "cv"]

    numeric_candidates: set[str] = set()
    for col in base_numeric:
        if col in dataset.columns:
            numeric_candidates.add(col)

    # add aggregated stats
    for prefix in stat_prefixes:
        for stat in stats:
            col_name = f"{prefix}_{stat}"
            if col_name in dataset.columns:
                numeric_candidates.add(col_name)

    # style counts and ratios
    style_cols = [col for col in dataset.columns if col.startswith("style_")]
    numeric_candidates.update(style_cols)

    numeric_features = sorted(numeric_candidates)
    categorical_features = [
        "grade",
        "track",
        "category",
    ]
    return numeric_features, categorical_features


def build_feature_pipeline(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_features)),
            ("cat", categorical_transformer, list(categorical_features)),
        ]
    )
    model = HistGradientBoostingClassifier(random_state=42)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def extract_feature_importance(
    pipeline: Pipeline,
    categorical_features: Iterable[str],
) -> List[Tuple[str, float]]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocess"]
    model: HistGradientBoostingClassifier = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return []
    feature_names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)  # type: ignore[arg-type]
        elif name == "cat":
            encoder: OneHotEncoder = transformer.named_steps["encoder"]  # type: ignore[assignment]
            encoder_features = encoder.get_feature_names_out(cols)  # type: ignore[arg-type]
            feature_names.extend(encoder_features.tolist())
    importances = model.feature_importances_
    return sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)


def train_model(
    dataset: pd.DataFrame,
    payout_threshold: int,
    test_size: float,
    random_state: int,
) -> Dict[str, object]:
    numeric_features, categorical_features = select_feature_columns(dataset)
    required_columns = (
        ["target_high_payout"]
        + numeric_features
        + categorical_features
    )
    missing_cols = [col for col in required_columns if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    working_df = dataset.dropna(subset=["target_high_payout"])
    X = working_df[numeric_features + categorical_features]
    y = working_df["target_high_payout"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    pipeline = build_feature_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = np.nan
    report = classification_report(y_test, y_pred, output_dict=True)
    feature_importances = extract_feature_importance(pipeline, categorical_features)
    return {
        "pipeline": pipeline,
        "report": report,
        "roc_auc": roc_auc,
        "payout_threshold": payout_threshold,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_importances": feature_importances,
    }


def save_metrics(result: Dict[str, object]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = result["report"]
    roc_auc = result["roc_auc"]
    payout_threshold = result["payout_threshold"]
    feature_importances = result["feature_importances"]
    summary = {
        "payout_threshold": payout_threshold,
        "roc_auc": roc_auc,
        "classification_report": report,
        "top_features": feature_importances[:20],
    }
    METRICS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-payout prediction model.")
    parser.add_argument(
        "--results",
        default=DATA_DIR / "keirin_results_20240101_20251004.csv",
        type=Path,
        help="Path to aggregated results CSV.",
    )
    parser.add_argument(
        "--prerace",
        default=DATA_DIR / "keirin_prerace_20240101_20251004.csv",
        type=Path,
        help="Path to prerace CSV.",
    )
    parser.add_argument(
        "--entries",
        default=DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv",
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
        "--test-size",
        default=0.2,
        type=float,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip persisting trained model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.results, args.threshold)
    prerace = load_prerace(args.prerace)
    entries = aggregate_entries(args.entries)
    dataset = merge_datasets(results, prerace, entries)
    dataset = add_derived_features(dataset)
    training_result = train_model(
        dataset=dataset,
        payout_threshold=args.threshold,
        test_size=args.test_size,
        random_state=args.seed,
    )
    if not args.no_save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(training_result["pipeline"], MODEL_PATH)
    save_metrics(training_result)
    report_summary = training_result["report"]["weighted avg"]  # type: ignore[index]
    print(json.dumps(
        {
            "roc_auc": training_result["roc_auc"],
            "weighted_precision": report_summary["precision"],
            "weighted_recall": report_summary["recall"],
            "weighted_f1": report_summary["f1-score"],
            "payout_threshold": args.threshold,
            "model_path": None if args.no_save_model else str(MODEL_PATH),
            "metrics_path": str(METRICS_PATH),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
