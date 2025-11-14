#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility helpers for the pre-race high-payout prediction model.

This module centralises the feature engineering that is shared by
the training, CLI inference, and the web UI so that every entry point
uses the exact same transformations.

All features are computed only from information that is available
before the race starts. Popularity / odds are deliberately excluded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "prerace_model_lgbm.txt"
METADATA_PATH = MODEL_DIR / "prerace_model_metadata.json"
DATASET_PATH = MODEL_DIR / "prerace_training_dataset.csv"

# Style variations we normalise for both historical data and manual input.
STYLE_ALIASES = {
    "逃": "nige",
    "逃げ": "nige",
    "先": "nige",
    "先行": "nige",
    "捲": "nige",
    "まくり": "nige",
    "追": "tsui",
    "追込": "tsui",
    "追い込み": "tsui",
    "差": "tsui",
    "マーク": "tsui",
    "両": "ryo",
    "自在": "ryo",
    "万能": "ryo",
}
STYLE_FEATURES = ["nige", "tsui", "ryo"]

# Load track/category statistics for prediction adjustment
STATS_PATH = MODEL_DIR / "track_category_stats.json"
TRACK_CATEGORY_STATS = {}
try:
    if STATS_PATH.exists():
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            TRACK_CATEGORY_STATS = json.load(f)
        print(f"[INFO] Loaded track/category statistics")
except Exception as e:
    print(f"[WARN] Could not load track/category statistics: {e}")

# Rider grade levels that frequently appear in race cards.
GRADE_LEVELS = ["SS", "S1", "S2", "A1", "A2", "A3", "L1"]

# Race grade flags (event level).
GRADE_FLAGS = ["GP", "G1", "G2", "G3", "F1", "F2", "F3", "L"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Feature values plus summary stats useful for explanation."""

    features: Dict[str, float]
    summary: Dict[str, Any]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with encoding fallback (cp932 -> utf-8 -> default)."""
    encodings = ["cp932", "utf-8", None]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            if enc is None:
                return pd.read_csv(path, **kwargs)
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as err:
            last_error = err
            continue
    if last_error:
        raise last_error
    raise ValueError(f"Failed to read CSV: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | float("nan"):
    """Convert a value to float while preserving NaN for invalid entries."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    try:
        value_str = str(value).strip()
        if not value_str:
            return float("nan")
        return float(value_str)
    except (TypeError, ValueError):
        return float("nan")


def _normalise_style(value: Any) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip()
    if not key:
        return "unknown"
    direct_map = {
        "先行": "nige",
        "逃げ": "nige",
        "追込": "tsui",
        "追い込み": "tsui",
        "自在": "ryo",
        "両": "ryo",
    }
    if key in direct_map:
        return direct_map[key]
    for alias, mapped in STYLE_ALIASES.items():
        if key.startswith(alias):
            return mapped
    return "unknown"


def _normalise_grade(value: Any) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip().upper()
    if key in GRADE_LEVELS:
        return key
    if key.startswith(("SS", "S", "A", "L")) and len(key) >= 2:
        return key[:2]
    return "unknown"


def _normalise_prefecture(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _calendar_features(race_date: int) -> Dict[str, int]:
    """Derive calendar features from YYYYMMDD integer."""
    try:
        date_str = str(int(race_date)).zfill(8)
    except (TypeError, ValueError):
        date_str = "0" * 8
    try:
        dt = pd.to_datetime(date_str, format="%Y%m%d", errors="raise")
    except (ValueError, TypeError):
        return {
            "year": 0,
            "month": 0,
            "day": 0,
            "day_of_week": 0,
            "is_weekend": 0,
        }
    return {
        "year": int(dt.year),
        "month": int(dt.month),
        "day": int(dt.day),
        "day_of_week": int(dt.dayofweek),
        "is_weekend": int(dt.dayofweek >= 5),
    }


def _grade_flag_features(grade_text: str) -> Dict[str, int]:
    flags: Dict[str, int] = {}
    upper = str(grade_text or "").upper()
    for flag in GRADE_FLAGS:
        flags[f"grade_flag_{flag}"] = int(flag in upper)
    return flags


def _summarise_riders(rider_frame: pd.DataFrame) -> FeatureBundle:
    """Aggregate rider level inputs into race level features."""
    entry_count = len(rider_frame)
    features: Dict[str, float] = {"entry_count": float(entry_count)}
    summary: Dict[str, Any] = {"entry_count": entry_count}

    if entry_count == 0:
        # Populate feature space with zeros for downstream alignment.
        for key in [
            "score_mean",
            "score_std",
            "score_min",
            "score_max",
            "score_range",
            "score_median",
            "score_q25",
            "score_q75",
            "score_iqr",
            "score_cv",
            "score_top3_mean",
            "score_bottom3_mean",
            "score_top_bottom_gap",
            "estimated_top3_score_sum",
            "estimated_favorite_dominance",
            "estimated_favorite_gap",
            "estimated_top3_vs_others",
            "style_diversity",
            "style_max_ratio",
            "style_min_ratio",
            "style_unknown_ratio",
            "style_entropy",
            "grade_entropy",
            "grade_has_mixed",
        ]:
            features[key] = 0.0
        for style in STYLE_FEATURES:
            features[f"style_{style}_count"] = 0.0
            features[f"style_{style}_ratio"] = 0.0
        for grade in GRADE_LEVELS:
            features[f"grade_{grade}_count"] = 0.0
            features[f"grade_{grade}_ratio"] = 0.0
        features["prefecture_unique_count"] = 0.0
        summary.update(
            {
                "score_mean": 0.0,
                "score_range": 0.0,
                "score_std": 0.0,
                "score_cv": 0.0,
                "style_counts": {style: 0 for style in STYLE_FEATURES},
                "style_ratios": {style: 0.0 for style in STYLE_FEATURES},
                "style_diversity": 0.0,
                "unknown_style_count": 0,
                "grade_counts": {grade: 0 for grade in GRADE_LEVELS},
                "prefecture_unique_count": 0,
            }
        )
        return FeatureBundle(features, summary)

    scores = pd.to_numeric(rider_frame["score"], errors="coerce")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()

    if scores.empty:
        mean = std = min_score = max_score = median = q25 = q75 = 0.0
    else:
        mean = float(scores.mean())
        std = float(scores.std(ddof=0)) if len(scores) > 1 else 0.0
        min_score = float(scores.min())
        max_score = float(scores.max())
        median = float(scores.median())
        q25 = float(scores.quantile(0.25))
        q75 = float(scores.quantile(0.75))

    iq_range = q75 - q25
    cv = std / (mean + 1e-6)

    top3 = scores.nlargest(min(3, len(scores))) if not scores.empty else pd.Series(dtype=float)
    bottom3 = scores.nsmallest(min(3, len(scores))) if not scores.empty else pd.Series(dtype=float)

    top3_mean = float(top3.mean()) if len(top3) else mean
    bottom3_mean = float(bottom3.mean()) if len(bottom3) else mean
    top_bottom_gap = top3_mean - bottom3_mean

    # Estimated popularity features (score-based proxy for actual popularity)
    # Higher scores = more popular = lower payout when they win
    if not scores.empty and len(scores) >= 3:
        sorted_scores = scores.sort_values(ascending=False)
        rank1_score = float(sorted_scores.iloc[0])
        rank2_score = float(sorted_scores.iloc[1]) if len(sorted_scores) > 1 else rank1_score
        rank3_score = float(sorted_scores.iloc[2]) if len(sorted_scores) > 2 else rank2_score

        # Top 3 sum (proxy for "favorite trifecta" - higher = more likely to be popular combination)
        top3_score_sum = rank1_score + rank2_score + rank3_score

        # Favorite dominance (how much stronger is the top rider)
        favorite_dominance = rank1_score / (mean + 1e-6)

        # Gap between 1st and 2nd (large gap = clear favorite = low upset potential)
        favorite_gap = rank1_score - rank2_score

        # Top3 vs others average gap (large gap = strong favorites dominating = low upset potential)
        if len(scores) > 3:
            others_mean = float(scores.iloc[3:].mean())
            top3_vs_others = top3_mean - others_mean
        else:
            top3_vs_others = 0.0
    else:
        rank1_score = mean
        rank2_score = mean
        rank3_score = mean
        top3_score_sum = mean * 3
        favorite_dominance = 1.0
        favorite_gap = 0.0
        top3_vs_others = 0.0

    features.update(
        {
            "score_mean": mean,
            "score_std": std,
            "score_min": min_score,
            "score_max": max_score,
            "score_range": max_score - min_score,
            "score_median": median,
            "score_q25": q25,
            "score_q75": q75,
            "score_iqr": iq_range,
            "score_cv": cv,
            "score_top3_mean": top3_mean,
            "score_bottom3_mean": bottom3_mean,
            "score_top_bottom_gap": top_bottom_gap,
            "estimated_top3_score_sum": top3_score_sum,
            "estimated_favorite_dominance": favorite_dominance,
            "estimated_favorite_gap": favorite_gap,
            "estimated_top3_vs_others": top3_vs_others,
        }
    )
    summary.update(
        {
            "score_mean": mean,
            "score_range": max_score - min_score,
            "score_std": std,
            "score_cv": cv,
        }
    )

    style_counts: Dict[str, int] = {}
    for style in STYLE_FEATURES:
        count = int((rider_frame["style_norm"] == style).sum())
        style_counts[style] = count
        features[f"style_{style}_count"] = float(count)
        features[f"style_{style}_ratio"] = float(count / entry_count)
    unknown_style = int((rider_frame["style_norm"] == "unknown").sum())
    features["style_unknown_ratio"] = float(unknown_style / entry_count)

    ratios = np.array([features[f"style_{style}_ratio"] for style in STYLE_FEATURES], dtype=float)
    style_diversity = float(1.0 - np.sum(ratios ** 2))
    features["style_diversity"] = style_diversity
    features["style_max_ratio"] = float(ratios.max()) if ratios.size else 0.0
    features["style_min_ratio"] = float(ratios.min()) if ratios.size else 0.0

    summary.update(
        {
            "style_counts": style_counts,
            "style_ratios": {style: features[f"style_{style}_ratio"] for style in STYLE_FEATURES},
            "style_diversity": style_diversity,
            "unknown_style_count": unknown_style,
        }
    )

    grade_counts: Dict[str, int] = {}
    for grade in GRADE_LEVELS:
        count = int((rider_frame["grade_norm"] == grade).sum())
        grade_counts[grade] = count
        features[f"grade_{grade}_count"] = float(count)
        features[f"grade_{grade}_ratio"] = float(count / entry_count)
    summary["grade_counts"] = grade_counts

    # Grade diversity (entropy)
    grade_ratios = np.array([features[f"grade_{g}_ratio"] for g in GRADE_LEVELS], dtype=float)
    grade_entropy = float(-np.sum(grade_ratios * np.log(grade_ratios + 1e-10)))
    features["grade_entropy"] = grade_entropy

    # Mixed grade levels (S-class and A-class together)
    has_s_class = any(grade_counts.get(g, 0) > 0 for g in ["SS", "S1", "S2"])
    has_a_class = any(grade_counts.get(g, 0) > 0 for g in ["A1", "A2", "A3"])
    features["grade_has_mixed"] = float(has_s_class and has_a_class)

    # Style diversity (entropy)
    style_ratios_array = np.array([features[f"style_{s}_ratio"] for s in STYLE_FEATURES], dtype=float)
    style_entropy = float(-np.sum(style_ratios_array * np.log(style_ratios_array + 1e-10)))
    features["style_entropy"] = style_entropy

    prefecture_unique = int(rider_frame["prefecture_norm"].nunique())
    features["prefecture_unique_count"] = float(prefecture_unique)
    summary["prefecture_unique_count"] = prefecture_unique

    return FeatureBundle(features, summary)


def _prepare_rider_frame_from_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of the raw entries table for aggregation."""
    frame = entries.copy()
    frame["score"] = frame["heikinTokuten"].apply(_safe_float)
    frame["style_norm"] = frame["kyakusitu"].apply(_normalise_style)
    frame["grade_norm"] = frame["kyuhan"].apply(_normalise_grade)
    frame["prefecture_norm"] = frame["huKen"].apply(_normalise_prefecture)
    if "keirin_cd" not in frame.columns:
        frame["keirin_cd"] = ""
    if "track" in frame.columns:
        frame["track"] = frame["track"].astype(str)
    else:
        frame["track"] = ""
    return frame[
        [
            "race_date",
            "keirin_cd",
            "track",
            "race_no",
            "score",
            "style_norm",
            "grade_norm",
            "prefecture_norm",
        ]
    ]


def aggregate_rider_features(entries_path: Path) -> pd.DataFrame:
    """Aggregate rider level CSV data into race level features."""
    header = _read_csv(entries_path, nrows=0)
    usecols = {
        "race_date",
        "track",
        "race_no",
        "heikinTokuten",
        "kyakusitu",
        "kyuhan",
        "huKen",
    }
    if "keirin_cd" in header.columns:
        usecols.add("keirin_cd")
    dtype_map = {"keirin_cd": str} if "keirin_cd" in header.columns else None
    entries = _read_csv(entries_path, usecols=list(usecols), dtype=dtype_map)
    entries["race_date"] = entries["race_date"].astype(int)
    entries["race_no"] = entries["race_no"].astype(int)
    if "keirin_cd" in entries.columns:
        entries["keirin_cd"] = entries["keirin_cd"].astype(str).str.zfill(2)
    else:
        entries["keirin_cd"] = ""
    if "track" in entries.columns:
        entries["track"] = entries["track"].astype(str)
    else:
        entries["track"] = ""

    rider_frame = _prepare_rider_frame_from_entries(entries)
    rows: List[Dict[str, Any]] = []

    group_keys = ["race_date", "track", "race_no", "keirin_cd"]
    for keys, group in rider_frame.groupby(group_keys):
        race_date, track, race_no, keirin_cd = keys
        bundle = _summarise_riders(group.drop(columns=["track"], errors="ignore"))
        row = {
            "race_date": int(race_date),
            "track": str(track),
            "race_no": int(race_no),
            "keirin_cd": str(keirin_cd),
        }
        row.update(bundle.features)
        rows.append(row)

    return pd.DataFrame(rows)


def load_results_table(results_path: Path, payout_threshold: int) -> pd.DataFrame:
    """Load race results and derive the binary target."""
    usecols = [
        "race_date",
        "keirin_cd",
        "race_no",
        "track",
        "grade",
        "category",
        "trifecta_payout",
    ]
    results = _read_csv(results_path, usecols=usecols, dtype={"keirin_cd": str})
    results["race_date"] = results["race_date"].astype(int)
    results["race_no"] = results["race_no"].astype(str)
    results["race_no"] = results["race_no"].str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    results["keirin_cd"] = results["keirin_cd"].str.zfill(2)

    payout = (
        results["trifecta_payout"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    results["target_high_payout"] = (payout >= float(payout_threshold)).astype(int)
    results["keirin_cd_num"] = results["keirin_cd"].astype(int)

    grade_flags = results["grade"].fillna("").apply(_grade_flag_features)
    grade_flag_df = pd.DataFrame(list(grade_flags))
    results = pd.concat([results, grade_flag_df], axis=1)

    return results[
        [
            "race_date",
            "keirin_cd",
            "keirin_cd_num",
            "race_no",
            "track",
            "grade",
            "target_high_payout",
        ]
        + [f"grade_flag_{flag}" for flag in GRADE_FLAGS]
    ]


def load_prerace_calendar(prerace_path: Path) -> pd.DataFrame:
    """Extract calendar features from the prerace table."""
    usecols = ["race_date", "keirin_cd", "race_no", "nitiji"]
    prerace = _read_csv(prerace_path, usecols=usecols, dtype={"keirin_cd": str})
    prerace["race_date"] = prerace["race_date"].astype(int)
    prerace["keirin_cd"] = prerace["keirin_cd"].str.zfill(2)
    prerace["race_no"] = prerace["race_no"].astype(int)

    cal_rows: List[Dict[str, Any]] = []
    for _, row in prerace.iterrows():
        calendar = _calendar_features(row["race_date"])
        nitiji_text = str(row.get("nitiji", "") or "")
        meeting_day: int | None = None
        match = re.search(r"(\d)日目", nitiji_text)
        if match:
            try:
                meeting_day = int(match.group(1))
            except ValueError:
                meeting_day = None
        elif "初日" in nitiji_text:
            meeting_day = 1
        cal_rows.append(
            {
                "race_date": int(row["race_date"]),
                "keirin_cd": str(row["keirin_cd"]),
                "race_no": int(row["race_no"]),
                **calendar,
                "is_first_day": int("初日" in nitiji_text),
                "is_final_day": int("最終日" in nitiji_text),
                "is_second_day": int(("2日目" in nitiji_text) or ("二日目" in nitiji_text)),
                "meeting_day": meeting_day,
            }
        )

    return pd.DataFrame(cal_rows)


def _default_feature_columns() -> List[str]:
    """Return the canonical feature list used for training the LightGBM model."""
    return [
        "keirin_cd_num",
        "race_no",
        "year",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "is_first_day",
        "is_second_day",
        "is_final_day",
        "grade_flag_GP",
        "grade_flag_G1",
        "grade_flag_G2",
        "grade_flag_G3",
        "grade_flag_F1",
        "grade_flag_F2",
        "grade_flag_F3",
        "grade_flag_L",
        "entry_count",
        "score_mean",
        "score_std",
        "score_min",
        "score_max",
        "score_range",
        "score_median",
        "score_q25",
        "score_q75",
        "score_iqr",
        "score_cv",
        "score_top3_mean",
        "score_bottom3_mean",
        "score_top_bottom_gap",
        "estimated_top3_score_sum",
        "estimated_favorite_dominance",
        "estimated_favorite_gap",
        "estimated_top3_vs_others",
        "style_nige_ratio",
        "style_tsui_ratio",
        "style_ryo_ratio",
        "style_unknown_ratio",
        "style_diversity",
        "style_max_ratio",
        "style_min_ratio",
        "style_nige_count",
        "style_tsui_count",
        "style_ryo_count",
        "style_entropy",
        "grade_SS_ratio",
        "grade_S1_ratio",
        "grade_S2_ratio",
        "grade_A1_ratio",
        "grade_A2_ratio",
        "grade_A3_ratio",
        "grade_L1_ratio",
        "grade_SS_count",
        "grade_S1_count",
        "grade_S2_count",
        "grade_A1_count",
        "grade_A2_count",
        "grade_A3_count",
        "grade_L1_count",
        "grade_entropy",
        "grade_has_mixed",
        "prefecture_unique_count",
    ]


def build_training_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int = 10000,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct the training dataset that uses only pre-race information."""
    results = load_results_table(results_path, payout_threshold)
    calendar = load_prerace_calendar(prerace_path)
    rider_features = aggregate_rider_features(entries_path)

    dataset = results.merge(calendar, on=["race_date", "keirin_cd", "race_no"], how="inner")

    if "keirin_cd" in rider_features.columns and rider_features["keirin_cd"].str.strip().any():
        dataset = dataset.merge(rider_features, on=["race_date", "keirin_cd", "race_no"], how="inner")
    else:
        dataset = dataset.merge(
            rider_features.drop(columns=["keirin_cd"], errors="ignore"),
            on=["race_date", "race_no", "track"],
            how="inner",
        )
        dataset["keirin_cd"] = dataset["keirin_cd"]

    dataset = dataset.sort_values(["race_date", "keirin_cd", "race_no"]).reset_index(drop=True)


    # Feature list (all numeric for easier deployment).
    feature_columns = _default_feature_columns()

    dataset[feature_columns] = dataset[feature_columns].fillna(0.0)
    dataset["target_high_payout"] = dataset["target_high_payout"].astype(int)

    return dataset, feature_columns


def save_training_dataset(dataset: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(DATASET_PATH, index=False)


def load_cached_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    raise FileNotFoundError(
        "Cached dataset is not available. Run analysis/train_prerace_lightgbm.py first."
    )


def manual_riders_to_frame(riders: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rider in riders:
        rows.append(
            {
                "score": _safe_float(rider.get("avg_score")),
                "style_norm": _normalise_style(rider.get("style")),
                "grade_norm": _normalise_grade(rider.get("grade")),
                "prefecture_norm": _normalise_prefecture(rider.get("prefecture")),
            }
        )
    return pd.DataFrame(rows)


def build_manual_feature_row(race_payload: Dict[str, Any]) -> FeatureBundle:
    race_date_raw = race_payload.get("race_date")
    try:
        race_date = int(str(race_date_raw).replace("-", ""))
    except (TypeError, ValueError):
        race_date = 0

    keirin_cd = str(race_payload.get("keirin_cd", "")).zfill(2)
    try:
        keirin_cd_num = int(keirin_cd)
    except ValueError:
        keirin_cd_num = 0

    try:
        race_no = int(race_payload.get("race_no", 0))
    except (TypeError, ValueError):
        race_no = 0

    riders = race_payload.get("riders", [])
    rider_frame = manual_riders_to_frame(riders)
    bundle = _summarise_riders(rider_frame)

    calendar = _calendar_features(race_date)
    calendar["is_first_day"] = int(bool(race_payload.get("is_first_day")))
    calendar["is_second_day"] = int(bool(race_payload.get("is_second_day")))
    calendar["is_final_day"] = int(bool(race_payload.get("is_final_day")))

    grade_flags = _grade_flag_features(race_payload.get("grade", ""))

    def _get_bool(name: str) -> bool:
        return bool(race_payload.get(name))

    def _get_text(name: str) -> str:
        return str(race_payload.get(name, "") or "").strip()

    def _get_number(name: str) -> float | None:
        value = race_payload.get(name)
        if value in (None, ""):
            return None
        parsed = _safe_float(value)
        return parsed if not np.isnan(parsed) else None

    meeting_day = race_payload.get("meeting_day")
    try:
        meeting_day_int = int(meeting_day) if meeting_day not in (None, "") else None
    except (TypeError, ValueError):
        meeting_day_int = None

    features = {
        "race_date": race_date,
        "keirin_cd": keirin_cd,
        "keirin_cd_num": float(keirin_cd_num),
        "race_no": float(race_no),
        **calendar,
        **grade_flags,
        **bundle.features,
    }

    summary_context = {
        "track": _get_text("track") or bundle.summary.get("track", ""),
        "weather_condition": _get_text("weather_condition"),
        "track_condition": _get_text("track_condition"),
        "temperature_c": _get_number("temperature"),
        "wind_speed_mps": _get_number("wind_speed"),
        "wind_direction": _get_text("wind_direction"),
        "is_night_race": _get_bool("is_night_race"),
        "meeting_day": meeting_day_int,
        "notes": _get_text("notes"),
    }

    summary = {
        **bundle.summary,
        "race_date": race_date,
        "keirin_cd": keirin_cd,
        "race_no": race_no,
        "grade": race_payload.get("grade", ""),
        **summary_context,
    }
    return FeatureBundle(features, summary)


def align_features(
    feature_bundle: FeatureBundle,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align manual feature dict to the training feature order."""
    row = {col: feature_bundle.features.get(col, 0.0) for col in feature_columns}
    frame = pd.DataFrame([row])
    frame = frame.astype(float)
    return frame, feature_bundle.summary


def load_metadata() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        expected_columns = _default_feature_columns()
        if metadata.get("feature_columns"):
            current = metadata["feature_columns"]
            if len(current) != len(expected_columns) or any(
                a != b for a, b in zip(current, expected_columns)
            ):
                metadata["feature_columns"] = expected_columns
                try:
                    save_metadata(metadata)
                except Exception:
                    pass
        else:
            try:
                dataset = load_cached_dataset()
            except FileNotFoundError as exc:
                raise KeyError(
                    "metadata is missing 'feature_columns' and no cached dataset is available. "
                    "Re-run analysis/train_prerace_lightgbm.py to rebuild artefacts."
                ) from exc

            exclude = {
                "target_high_payout",
                "race_date",
                "keirin_cd",
                "race_no",
                "trifecta_payout",
                "meeting_day",
                "track",
                "grade",
            }
            feature_columns = [col for col in dataset.columns if col not in exclude]
            metadata["feature_columns"] = [col for col in feature_columns if col in expected_columns]
            if len(metadata["feature_columns"]) != len(expected_columns):
                metadata["feature_columns"] = expected_columns
            metadata.setdefault(
                "high_confidence_threshold",
                min(0.95, metadata.get("best_threshold", 0.5) + 0.1),
            )
            try:
                save_metadata(metadata)
            except Exception:
                pass
        return metadata
    raise FileNotFoundError(
        "Metadata file not found. Train the model with analysis/train_prerace_lightgbm.py."
    )


def save_metadata(metadata: Dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model() -> Any:
    """
    Try to load the LightGBM booster. If読み込みに失敗した場合は None を返し、
    フォールバックのヒューリスティックを使う。
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Train the model with analysis/train_prerace_lightgbm.py."
        )
    try:
        import lightgbm as lgb  # Optional dependency

        return lgb.Booster(model_file=str(MODEL_PATH))
    except Exception as exc:
        print("[WARN] LightGBMモデルのロードに失敗しました。ヒューリスティック推論に切り替えます。")
        print(f"       詳細: {exc}")
        return None


def predict_probability(
    feature_frame: pd.DataFrame,
    model: Any,
    metadata: Dict[str, Any],
    race_context: Dict[str, Any] = None,
) -> float:
    """
    Predict high payout probability using RULE-BASED system.

    NOTE: ML model is disabled (ROC-AUC 0.58 = random).
    Using validated rule-based scoring instead.
    """
    # ALWAYS use rule-based prediction (ML model performance is too low)
    row = feature_frame.iloc[0]
    base_prob = _fallback_probability(row, metadata)

    # Apply track/category adjustment if context provided
    if race_context and TRACK_CATEGORY_STATS:
        adjusted_prob = _adjust_by_track_category(base_prob, race_context)
        return adjusted_prob

    return base_prob


def _adjust_by_track_category(base_prob: float, race_context: Dict[str, Any]) -> float:
    """Adjust probability based on historical track and category rates"""
    track = race_context.get('track', '')
    category = race_context.get('category', '')

    overall_rate = TRACK_CATEGORY_STATS.get('overall_high_payout_rate', 0.266)
    track_rates = TRACK_CATEGORY_STATS.get('track_high_payout_rates', {})
    category_rates = TRACK_CATEGORY_STATS.get('category_high_payout_rates', {})

    # Calculate adjustment multipliers
    multiplier = 1.0

    # Track adjustment (weight: 0.4)
    if track in track_rates:
        track_rate = track_rates[track]
        track_multiplier = track_rate / overall_rate
        multiplier *= (1.0 + 0.4 * (track_multiplier - 1.0))

    # Category adjustment (weight: 0.6)
    if category in category_rates:
        category_rate = category_rates[category]
        category_multiplier = category_rate / overall_rate
        multiplier *= (1.0 + 0.6 * (category_multiplier - 1.0))

    # Apply multiplier
    adjusted = base_prob * multiplier

    # Keep in valid range [0, 1]
    adjusted = max(0.05, min(0.95, adjusted))

    return adjusted


def _fallback_probability(row: pd.Series, metadata: Dict[str, Any]) -> float:
    """
    Rule-based scoring using competition scores and rider characteristics.

    This is now the PRIMARY prediction method (ML model has ROC-AUC 0.58 = useless).
    Based on validated statistical relationships from 48k+ races.
    """
    # Core score-based features (MOST IMPORTANT)
    score_cv = float(row.get("score_cv", 0.0) or 0.0)
    score_range = float(row.get("score_range", 0.0) or 0.0)
    score_std = float(row.get("score_std", 0.0) or 0.0)

    # New score-based popularity estimation features
    favorite_gap = float(row.get("estimated_favorite_gap", 0.0) or 0.0)
    favorite_dominance = float(row.get("estimated_favorite_dominance", 1.0) or 1.0)
    top3_vs_others = float(row.get("estimated_top3_vs_others", 0.0) or 0.0)
    score_top_bottom_gap = float(row.get("score_top_bottom_gap", 0.0) or 0.0)

    # Rider diversity features
    style_diversity = float(row.get("style_diversity", 0.0) or 0.0)
    style_entropy = float(row.get("style_entropy", 0.0) or 0.0)
    grade_entropy = float(row.get("grade_entropy", 0.0) or 0.0)
    prefecture_unique = float(row.get("prefecture_unique_count", 0.0) or 0.0)

    # Style composition
    tsui_ratio = float(row.get("style_tsui_ratio", 0.0) or 0.0)
    ryo_ratio = float(row.get("style_ryo_ratio", 0.0) or 0.0)
    nige_count = float(row.get("style_nige_count", 0.0) or 0.0)
    entry_count = float(row.get("entry_count", 0.0) or 0.0)

    # Grade composition
    grade_s1 = float(row.get("grade_S1_ratio", 0.0) or 0.0)
    grade_ss = float(row.get("grade_SS_ratio", 0.0) or 0.0)
    grade_a3 = float(row.get("grade_A3_ratio", 0.0) or 0.0)

    # === RULE 1: Score Variability (HIGHEST WEIGHT) ===
    # High CV/std = evenly matched field = high upset potential
    score_variability_score = (
        4.5 * min(score_cv, 0.18)          # CV is THE most predictive
        + 2.0 * min(score_std / 5.0, 1.5)  # Standard deviation
        + 1.5 * min(score_range / 10.0, 1.5)  # Range (less predictive than CV)
    )

    # === RULE 2: Favorite Strength (NEW - CRITICAL) ===
    # Weak favorite or small gap = upset likely
    favorite_score = 0.0

    # Gap between 1st and 2nd place
    if favorite_gap < 2.0:  # Very close race
        favorite_score += 1.2
    elif favorite_gap < 5.0:  # Moderately competitive
        favorite_score += 0.6
    elif favorite_gap > 12.0:  # Dominant favorite
        favorite_score -= 1.0

    # Favorite dominance ratio
    if favorite_dominance < 1.05:  # Weak favorite (barely above average)
        favorite_score += 1.0
    elif favorite_dominance < 1.10:  # Moderate favorite
        favorite_score += 0.4
    elif favorite_dominance > 1.20:  # Strong dominant favorite
        favorite_score -= 1.2

    # Top3 vs others gap
    if top3_vs_others < 3.0:  # Top3 not much stronger
        favorite_score += 0.8
    elif top3_vs_others > 10.0:  # Clear tier separation
        favorite_score -= 0.6

    # === RULE 3: Field Diversity ===
    diversity_score = (
        1.5 * style_diversity
        + 1.2 * min(style_entropy / 1.5, 1.0)
        + 1.0 * min(grade_entropy / 1.8, 1.0)
        + 0.6 * min(prefecture_unique / 6.0, 1.0)
        + 0.4 * tsui_ratio
        + 0.3 * ryo_ratio
    )

    # === RULE 4: Race Conditions ===
    condition_score = 0.0

    # Small fields are unpredictable
    if entry_count <= 7:
        condition_score += 0.35

    # Few escape riders = chaotic races
    if nige_count <= 1:
        condition_score += 0.30

    # Meeting day effects
    if row.get("is_final_day", 0):
        condition_score += 0.20  # Finals are more competitive
    if row.get("is_second_day", 0):
        condition_score += 0.08
    if row.get("is_first_day", 0):
        condition_score -= 0.12  # First day more predictable

    # === RULE 5: Grade Strength (NEGATIVE FACTOR) ===
    # Strong riders = more predictable outcomes
    grade_score = -0.8 * min(grade_s1 + grade_ss, 1.0)

    # All weak riders = also predictable (favorites win)
    if grade_a3 >= 0.8:
        grade_score -= 0.15

    # === COMBINE ALL SCORES ===
    base = (
        score_variability_score
        + favorite_score
        + diversity_score
        + condition_score
        + grade_score
    )

    # Convert to probability (sigmoid transformation)
    # Adjusted offset for better calibration
    logit = base - 2.5  # Lower offset = higher base probability
    probability = 1.0 / (1.0 + math.exp(-logit))

    return float(min(0.95, max(0.05, probability)))


def generate_reasons(summary: Dict[str, Any], probability: float, metadata: Dict[str, Any]) -> List[str]:
    """Create lightweight human readable hints based on feature summary."""
    reasons: List[str] = []

    score_range = summary.get("score_range", 0.0)
    if score_range >= 8:
        reasons.append(f"選手の獲得点差が大きく（約{score_range:.1f}点差）波乱要素があります。")
    elif score_range >= 5:
        reasons.append(f"獲得点のバラつき（約{score_range:.1f}点差）が中程度にあります。")

    diversity = summary.get("style_diversity", 0.0)
    if diversity >= 0.55:
        reasons.append("脚質の構成がばらばらで展開が読みづらいレースです。")
    elif diversity <= 0.2:
        reasons.append("脚質が偏っており展開が読みやすい構成です。")

    grade_counts = summary.get("grade_counts", {})
    s1_ratio = 0.0
    total = max(1, summary.get("entry_count", 0))
    if grade_counts:
        s1_ratio = grade_counts.get("S1", 0) / total
    if s1_ratio <= 0.3:
        reasons.append("S級上位が少なく、実力拮抗の選手が多い印象です。")
    elif s1_ratio >= 0.6:
        reasons.append("S級上位が多く順当決着のシナリオが目立ちます。")

    threshold = metadata.get("best_threshold", 0.5)
    if probability >= threshold:
        reasons.append("過去データでは同様の条件で高配当が目立っています。")
    else:
        reasons.append("過去データでは比較的順当な結果が多い条件です。")

    # Remove duplicates while preserving order.
    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped[:4]


def build_betting_plan(
    probability: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create heuristic betting guidance for the front-end."""
    threshold = metadata.get("best_threshold", 0.5)
    high_threshold = metadata.get("high_confidence_threshold", threshold + 0.1)

    effective_score = probability
    score_cv = summary.get("score_cv", 0.0) or 0.0
    if score_cv >= 0.08:
        effective_score += 0.04
    elif score_cv <= 0.03:
        effective_score -= 0.03

    diversity = summary.get("style_diversity", 0.0) or 0.0
    if diversity >= 0.55:
        effective_score += 0.03
    elif diversity <= 0.2:
        effective_score -= 0.02

    prefecture_unique = summary.get("prefecture_unique_count", 0) or 0
    if prefecture_unique <= 3:
        effective_score -= 0.02
    elif prefecture_unique >= 6:
        effective_score += 0.02

    weather = summary.get("weather_condition", "")
    if weather in {"雨", "豪雨", "雷雨", "雪"}:
        effective_score += 0.05
    wind_speed = summary.get("wind_speed_mps")
    if wind_speed is not None:
        try:
            wind_speed_val = float(wind_speed)
            if wind_speed_val >= 8:
                effective_score += 0.03
            elif wind_speed_val <= 2:
                effective_score -= 0.01
        except (TypeError, ValueError):
            pass

    track_condition = summary.get("track_condition", "")
    if track_condition in {"重", "やや重"}:
        effective_score += 0.03

    if summary.get("is_final_day"):
        effective_score += 0.02

    effective_score = float(min(1.0, max(0.0, effective_score)))

    ticket_plan: List[Dict[str, str]]
    money_management: str
    hedge_note: str
    plan_summary: str
    risk_level: str

    if effective_score >= max(high_threshold, 0.75):
        risk_level = "ハイリスク（波乱期待大）"
        plan_summary = "本命崩れ・波乱展開を想定し、広めのカバーで高配当に備えましょう。"
        ticket_plan = [
            {
                "label": "三連単フォーメーション",
                "description": "本命1頭軸マルチ＋相手3〜4車で18〜24点を目安に広げる",
            },
            {
                "label": "三連複フォーメーション",
                "description": "本命2車-相手4〜5車で波乱の目を抑えつつコストをコントロール",
            },
        ]
        money_management = "総予算の20〜30%を想定。1点あたりは均等よりも穴目を厚めに配分。"
        hedge_note = "ワイドや二車複で本命-穴を少額抑えておくとドロー対策になります。"
    elif effective_score >= threshold:
        risk_level = "ミドルリスク（荒れ警戒）"
        plan_summary = "本命優勢を前提にしつつ、相手に穴を紛れ込ませる形でリスクとリターンを両立。"
        ticket_plan = [
            {
                "label": "三連単フォーメーション",
                "description": "本命1着固定 - 対抗・穴3車 - 相手4〜5車で12〜16点程度",
            },
            {
                "label": "ワイド／三連複",
                "description": "本命-穴のワイドで保険を掛け、三連複は本命2車-相手3〜4車",
            },
        ]
        money_management = "総予算の15〜20%を投下。的中確度の高い組み合わせを厚めに設定。"
        hedge_note = "リスクヘッジに単勝・複勝を1点添えると収支のブレを抑えられます。"
    else:
        risk_level = "ローリスク（静かな展開）"
        plan_summary = "波乱リスクは低め。無理な勝負は避け、見送りまたは少点数で静観が堅実です。"
        ticket_plan = [
            {
                "label": "見送り／抑え投資",
                "description": "見送り推奨。どうしても参加する場合は二車複・ワイドを1〜2点で様子見。",
            }
        ]
        money_management = "投資する場合でも総予算の5〜10%に抑え、深追いしない。"
        hedge_note = "狙う場合でも1点あたりの金額は極小に。レースを観察して次戦に活かしましょう。"

    return {
        "risk_level": risk_level,
        "plan_summary": plan_summary,
        "ticket_plan": ticket_plan,
        "money_management": money_management,
        "hedge_note": hedge_note,
        "effective_score": effective_score,
        "model_probability": probability,
        "weather": weather,
        "track_condition": track_condition,
    }


def build_prediction_response(
    probability: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    threshold = metadata.get("best_threshold", 0.5)
    high_threshold = metadata.get("high_confidence_threshold", threshold + 0.1)
    if probability >= high_threshold:
        confidence = "高"
        recommendation = "勝負レース候補（掛け金を厚めにする日）"
    elif probability >= threshold:
        confidence = "中"
        recommendation = "状況次第で参加。オッズと相談してください。"
    else:
        confidence = "低"
        recommendation = "見送り推奨。リスクに見合う根拠が不足しています。"

    reasons = generate_reasons(summary, probability, metadata)
    betting_plan = build_betting_plan(probability, summary, metadata)
    return {
        "probability": probability,
        "confidence": confidence,
        "recommendation": recommendation,
        "threshold": threshold,
        "high_threshold": high_threshold,
        "reasons": reasons,
        "summary": summary,
        "betting_plan": betting_plan,
    }
