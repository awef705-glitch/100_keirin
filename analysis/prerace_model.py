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

from analysis import betting_suggestions


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


# Regional line mapping (競輪のライン: 地域別共同作戦)
REGIONAL_LINES = {
    # 北日本ライン
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本",
    "秋田": "北日本", "山形": "北日本", "福島": "北日本",

    # 関東ライン
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東",
    "千葉": "関東", "東京": "関東", "神奈川": "関東", "山梨": "関東",

    # 北陸ライン
    "新潟": "北陸", "富山": "北陸", "石川": "北陸", "福井": "北陸",

    # 中部ライン
    "長野": "中部", "岐阜": "中部", "静岡": "中部", "愛知": "中部",

    # 近畿ライン
    "三重": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿",
    "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",

    # 中国ライン
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",

    # 四国ライン
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",

    # 九州ライン
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州",
    "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


def _get_regional_line(prefecture: str) -> str:
    """Map prefecture to regional line"""
    return REGIONAL_LINES.get(prefecture, "その他")


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
            "line_count",
            "dominant_line_ratio",
            "line_balance_std",
            "line_entropy",
            "line_score_gap",
            "back_count_mean",
            "back_count_std",
            "back_count_max",
            "back_count_top3_mean",
            "back_count_top3_sum",
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
                "back_count_mean": 0.0,
            }
        )
        return FeatureBundle(features, summary)

    scores = pd.to_numeric(rider_frame["score"], errors="coerce")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    
    back_counts = pd.to_numeric(rider_frame["back_count"], errors="coerce").fillna(0.0)

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

    # Back count features
    if not back_counts.empty:
        back_mean = float(back_counts.mean())
        back_std = float(back_counts.std(ddof=0)) if len(back_counts) > 1 else 0.0
        back_max = float(back_counts.max())
        
        # Top 3 riders by score - their B counts
        if not scores.empty:
            top3_indices = scores.nlargest(min(3, len(scores))).index
            back_top3 = back_counts.loc[top3_indices]
            back_top3_mean = float(back_top3.mean()) if not back_top3.empty else 0.0
            back_top3_sum = float(back_top3.sum())
        else:
            back_top3_mean = 0.0
            back_top3_sum = 0.0
    else:
        back_mean = 0.0
        back_std = 0.0
        back_max = 0.0
        back_top3_mean = 0.0
        back_top3_sum = 0.0

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
            "back_count_mean": back_mean,
            "back_count_std": back_std,
            "back_count_max": back_max,
            "back_count_top3_mean": back_top3_mean,
            "back_count_top3_sum": back_top3_sum,
        }
    )

    # Process tactical counts (nige, makuri, sasi, mark)
    for tact in ["nige", "makuri", "sasi", "mark"]:
        col = f"{tact}_count"
        vals = pd.to_numeric(rider_frame[col], errors="coerce").fillna(0.0)
        
        if not vals.empty:
            t_mean = float(vals.mean())
            t_max = float(vals.max())
            
            # Top 3 riders by score - their tactical counts
            if not scores.empty:
                top3_indices = scores.nlargest(min(3, len(scores))).index
                t_top3 = vals.loc[top3_indices]
                t_top3_sum = float(t_top3.sum())
            else:
                t_top3_sum = 0.0
        else:
            t_mean = 0.0
            t_max = 0.0
            t_top3_sum = 0.0
            
        features.update({
            f"{col}_mean": t_mean,
            f"{col}_max": t_max,
            f"{col}_top3_sum": t_top3_sum,
        })

    # Aggregate recent performance features
    for col in ["recent_win_rate", "recent_2ren_rate", "recent_3ren_rate"]:
        if col in rider_frame.columns:
            vals = pd.to_numeric(rider_frame[col], errors="coerce").fillna(0.0)
            if not vals.empty:
                features[f"{col}_mean"] = float(vals.mean())
                features[f"{col}_max"] = float(vals.max())
                features[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
            else:
                features[f"{col}_mean"] = 0.0
                features[f"{col}_max"] = 0.0
                features[f"{col}_std"] = 0.0
        else:
            features[f"{col}_mean"] = 0.0
            features[f"{col}_max"] = 0.0
            features[f"{col}_std"] = 0.0

    # Aggregate extra fields for rule-based logic
    for col in ["gear_ratio", "hs_count", "dq_points"]:
        if col in rider_frame.columns:
            vals = pd.to_numeric(rider_frame[col], errors="coerce").fillna(0.0)
            if not vals.empty:
                features[f"{col}_mean"] = float(vals.mean())
                features[f"{col}_max"] = float(vals.max())
            else:
                features[f"{col}_mean"] = 0.0
                features[f"{col}_max"] = 0.0
        else:
            features[f"{col}_mean"] = 0.0
            features[f"{col}_max"] = 0.0

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

    # === LINE ANALYSIS (ライン分析) ===
    # Map each rider to their regional line
    rider_frame["regional_line"] = rider_frame["prefecture_norm"].apply(_get_regional_line)

    # Count riders per line
    line_counts = rider_frame["regional_line"].value_counts().to_dict()

    # Calculate line balance metrics
    if len(line_counts) > 0:
        line_sizes = list(line_counts.values())
        max_line_size = max(line_sizes)
        min_line_size = min(line_sizes)
        line_count = len(line_sizes)

        # Dominant line ratio (largest line / total riders)
        dominant_line_ratio = max_line_size / entry_count if entry_count > 0 else 0.0

        # Line balance (std dev of line sizes)
        line_balance_std = float(np.std(line_sizes)) if len(line_sizes) > 1 else 0.0

        # Line entropy (how evenly distributed are riders across lines)
        line_ratios = np.array([c / entry_count for c in line_sizes], dtype=float)
        line_entropy = float(-np.sum(line_ratios * np.log(line_ratios + 1e-10)))
    else:
        max_line_size = 0
        min_line_size = 0
        line_count = 0
        dominant_line_ratio = 0.0
        line_balance_std = 0.0
        line_entropy = 0.0

    # Calculate average score per line (to detect dominant line)
    line_avg_scores = {}
    for line_name in line_counts.keys():
        line_riders = rider_frame[rider_frame["regional_line"] == line_name]
        line_scores = pd.to_numeric(line_riders["score"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(line_scores) > 0:
            line_avg_scores[line_name] = float(line_scores.mean())
        else:
            line_avg_scores[line_name] = 0.0

    # Score gap between strongest and weakest line
    if len(line_avg_scores) > 1:
        line_score_gap = max(line_avg_scores.values()) - min(line_avg_scores.values())
    else:
        line_score_gap = 0.0

    features["line_count"] = float(line_count)
    features["dominant_line_ratio"] = float(dominant_line_ratio)
    features["line_balance_std"] = float(line_balance_std)
    features["line_entropy"] = float(line_entropy)
    features["line_score_gap"] = float(line_score_gap)

    summary["line_counts"] = line_counts
    summary["line_avg_scores"] = line_avg_scores
    summary["dominant_line_ratio"] = dominant_line_ratio

    return FeatureBundle(features, summary)


def _prepare_rider_frame_from_entries(entries: pd.DataFrame) -> pd.DataFrame:
    """Rename columns of the raw entries table for aggregation."""
    frame = entries.copy()
    frame["score"] = frame["heikinTokuten"].apply(_safe_float)
    frame["back_count"] = frame["backCnt"].apply(_safe_float)
    frame["nige_count"] = frame["nigeCnt"].apply(_safe_float)
    frame["makuri_count"] = frame["makuriCnt"].apply(_safe_float)
    frame["sasi_count"] = frame["sasiCnt"].apply(_safe_float)
    frame["mark_count"] = frame["markCnt"].apply(_safe_float)
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
            "back_count",
            "nige_count",
            "makuri_count",
            "sasi_count",
            "mark_count",
            "style_norm",
            "grade_norm",
            "prefecture_norm",
            "recent_win_rate",
            "recent_2ren_rate",
            "recent_3ren_rate",
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
        "backCnt",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "kyakusitu",
        "kyuhan",
        "huKen",
    }
    if "keirin_cd" in header.columns:
        usecols.add("keirin_cd")
    usecols.add("sensyuName") # Needed for merging recent features
    dtype_map = {"keirin_cd": str} if "keirin_cd" in header.columns else None
    entries = _read_csv(entries_path, usecols=list(usecols), dtype=dtype_map)
    entries["race_date"] = entries["race_date"].astype(int)
    entries["race_no"] = entries["race_no"].astype(int)
    
    # Load recent features if available
    recent_features_path = Path("analysis/model_outputs/rider_recent_features.csv")
    if recent_features_path.exists():
        print(f"Loading recent features from {recent_features_path}...", flush=True)
        recent_df = pd.read_csv(recent_features_path)
        # Ensure types match for merge
        recent_df["race_date"] = recent_df["race_date"].astype(int)
        recent_df["race_no"] = recent_df["race_no"].astype(int)
        if "track" in recent_df.columns:
             recent_df["track"] = recent_df["track"].astype(str)
        
        # Merge
        # entries has 'track' (maybe), 'race_date', 'race_no', 'sensyuName'
        # recent_df has 'race_date', 'track', 'race_no', 'sensyuName', ...
        
        # Ensure entries has track as string
        if "track" in entries.columns:
            entries["track"] = entries["track"].astype(str)
        else:
            entries["track"] = ""
            
        entries = entries.merge(recent_df, on=["race_date", "track", "race_no", "sensyuName"], how="left")
        
        # Fill NA with 0
        entries["recent_win_rate"] = entries["recent_win_rate"].fillna(0.0)
        entries["recent_2ren_rate"] = entries["recent_2ren_rate"].fillna(0.0)
        entries["recent_3ren_rate"] = entries["recent_3ren_rate"].fillna(0.0)
    else:
        print("Recent features file not found. Skipping.", flush=True)
        entries["recent_win_rate"] = 0.0
        entries["recent_2ren_rate"] = 0.0
        entries["recent_3ren_rate"] = 0.0

    # Load track master to map track names to keirin_cd if needed
    if "keirin_cd" not in entries.columns or entries["keirin_cd"].isna().all():
        track_master_path = Path("analysis/model_outputs/track_master.json")
        if track_master_path.exists():
            import json
            with open(track_master_path, "r", encoding="utf-8") as f:
                track_master = json.load(f)
            # Create track name -> code mapping
            track_to_code = {track["name"]: track["code"] for track in track_master}
            # Map track names to keirin_cd
            if "track" in entries.columns:
                entries["keirin_cd"] = entries["track"].map(track_to_code).fillna("00")
            else:
                entries["keirin_cd"] = "00"
        else:
            entries["keirin_cd"] = "00"
    
    if "keirin_cd" in entries.columns:
        entries["keirin_cd"] = pd.to_numeric(entries["keirin_cd"], errors="coerce").fillna(0).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    else:
        entries["keirin_cd"] = "00"
        
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
    results["race_date"] = pd.to_numeric(results["race_date"], errors="coerce").fillna(0).astype(int)
    results["race_no"] = results["race_no"].astype(str)
    results["race_no"] = results["race_no"].str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    results["keirin_cd"] = pd.to_numeric(results["keirin_cd"], errors="coerce").fillna(0).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)

    payout = (
        results["trifecta_payout"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )
    results["target_high_payout"] = (payout >= float(payout_threshold)).astype(int)
    results["keirin_cd_num"] = pd.to_numeric(results["keirin_cd"], errors="coerce").fillna(0).astype(int)

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
    prerace["race_date"] = pd.to_numeric(prerace["race_date"], errors="coerce").fillna(0).astype(int)
    prerace["keirin_cd"] = pd.to_numeric(prerace["keirin_cd"], errors="coerce").fillna(0).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    prerace["race_no"] = pd.to_numeric(prerace["race_no"], errors="coerce").fillna(0).astype(int)

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
        "line_count",
        "dominant_line_ratio",
        "line_balance_std",
        "line_entropy",
        "line_score_gap",
        "back_count_mean",
        "back_count_std",
        "back_count_max",
        "back_count_top3_mean",
        "back_count_top3_sum",
        "nige_count_mean",
        "nige_count_max",
        "nige_count_top3_sum",
        "makuri_count_mean",
        "makuri_count_max",
        "makuri_count_top3_sum",
        "sasi_count_mean",
        "sasi_count_max",
        "sasi_count_top3_sum",
        "mark_count_mean",
        "mark_count_max",
        "mark_count_top3_sum",
        "recent_win_rate_mean",
        "recent_win_rate_max",
        "recent_win_rate_std",
        "recent_2ren_rate_mean",
        "recent_2ren_rate_max",
        "recent_2ren_rate_std",
        "recent_3ren_rate_mean",
        "recent_3ren_rate_max",
        "recent_3ren_rate_std",
    ]


def build_training_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int = 10000,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct the training dataset that uses only pre-race information."""
    print(f"Loading results from {results_path}...", flush=True)
    results = load_results_table(results_path, payout_threshold)
    print(f"Loaded {len(results)} results.", flush=True)
    
    print(f"Loading calendar from {prerace_path}...", flush=True)
    calendar = load_prerace_calendar(prerace_path)
    print(f"Loaded {len(calendar)} calendar entries.", flush=True)
    
    print(f"Loading rider features from {entries_path}...", flush=True)
    rider_features = aggregate_rider_features(entries_path)
    print(f"Loaded {len(rider_features)} rider feature rows.", flush=True)

    print("Merging results and calendar...", flush=True)
    dataset = results.merge(calendar, on=["race_date", "keirin_cd", "race_no"], how="inner")
    print(f"Merged dataset size: {len(dataset)}", flush=True)
    
    # Always merge using keirin_cd (track names may not match between datasets)
    print("Merging rider features...", flush=True)
    dataset = dataset.merge(rider_features, on=["race_date", "keirin_cd", "race_no"], how="inner")
    print(f"Final dataset size: {len(dataset)}", flush=True)

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
                "back_count": _safe_float(rider.get("back_count")),
                "nige_count": _safe_float(rider.get("nige_count")),
                "makuri_count": _safe_float(rider.get("makuri_count")),
                "sasi_count": _safe_float(rider.get("sasi_count")),
                "mark_count": _safe_float(rider.get("mark_count")),
                "style_norm": _normalise_style(rider.get("style")),
                "grade_norm": _normalise_grade(rider.get("grade")),
                "prefecture_norm": _normalise_prefecture(rider.get("prefecture")),
                "recent_win_rate": _safe_float(rider.get("recent_win_rate")),
                "recent_2ren_rate": _safe_float(rider.get("recent_2ren_rate")),
                "recent_3ren_rate": _safe_float(rider.get("recent_3ren_rate")),
                # Extra fields for rule-based logic
                "gear_ratio": _safe_float(rider.get("gear_ratio")),
                "hs_count": _safe_float(rider.get("hs_count")),
                "dq_points": _safe_float(rider.get("dq_points")),
                "home_bank": str(rider.get("home_bank", "")),
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
        "riders": riders,  # Add riders list for betting suggestions
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




def calculate_roughness_score(row: pd.Series, metadata: Dict[str, Any]) -> float:
    """
    Calculate 'Roughness Score' (荒れ度) on a 0-100 scale.
    0 = Extremely Stable (Favorite wins easily)
    100 = Extremely Chaotic (Anyone can win)
    
    This replaces the previous probability calibration to provide a more
    intuitive and sensitive metric for the user.
    """
    # Base score (Neutral race)
    score = 50.0

    # Core score-based features
    score_cv = float(row.get("score_cv", 0.0) or 0.0)
    favorite_gap = float(row.get("estimated_favorite_gap", 0.0) or 0.0)
    favorite_dominance = float(row.get("estimated_favorite_dominance", 1.0) or 1.0)
    
    # Line features
    dominant_line_ratio = float(row.get("dominant_line_ratio", 0.0) or 0.0)
    line_score_gap = float(row.get("line_score_gap", 0.0) or 0.0)

    # Tactical features
    nige_count = float(row.get("style_nige_count", 0.0) or 0.0)
    makuri_count = float(row.get("style_makuri_count", 0.0) or 0.0)
    
    # DEBUG PRINT
    # print(f"DEBUG: CV={score_cv:.4f}, Gap={favorite_gap:.1f}, Dom={dominant_line_ratio:.2f}, Nige={nige_count}")
    
    # === FACTOR 1: Score Variation (The most important factor) ===
    # High variation = Predictable (Strong vs Weak) -> Lower score
    # Low variation = Chaotic (All similar) -> Higher score
    
    # CV < 0.03 (Very tight) -> +25 points
    # CV > 0.10 (Very dispersed) -> -25 points
    if score_cv < 0.03:
        score += 25
    elif score_cv < 0.05:
        score += 15
    elif score_cv > 0.12:
        score -= 25
    elif score_cv > 0.08:
        score -= 15

    # === FACTOR 2: Favorite Strength ===
    # Gap between 1st and 2nd place scores
    # Gap < 2.0 (No clear favorite) -> +20 points
    # Gap > 15.0 (Dominant favorite) -> -20 points
    if favorite_gap < 2.0:
        score += 20
    elif favorite_gap < 5.0:
        score += 10
    elif favorite_gap > 15.0:
        score -= 20
    elif favorite_gap > 10.0:
        score -= 10

    # Favorite dominance ratio
    if favorite_dominance < 1.05:
        score += 15
    elif favorite_dominance > 1.15:
        score -= 15

    # === FACTOR 3: Line Balance ===
    # No dominant line (< 30%) -> Chaotic -> +10 points
    # One huge line (> 60%) -> Predictable -> -10 points
    if dominant_line_ratio < 0.3:
        score += 10
    elif dominant_line_ratio > 0.6:
        score -= 10
        
    # === FACTOR 4: Tactical Chaos ===
    # Too many 'Nige' (Runners) -> Pace becomes fast/chaotic -> +10
    # No 'Nige' -> Slow pace, hard to predict -> +5
    if nige_count >= 4:
        score += 10
    elif nige_count == 0:
        score += 5
        
    # === FACTOR 5: Grade Composition ===
    # All S-class -> High level battle -> Harder to predict? Or more stable?
    # Actually, mixed grades (S vs A) usually means S wins -> Stable -> Lower score
    has_mixed = float(row.get("grade_has_mixed", 0.0) or 0.0)
    if has_mixed > 0.5:
        score -= 15  # Mixed race is usually predictable (S beats A)

    # Clamp to 0-100
    return max(0.0, min(100.0, score))


def generate_reasons(summary: Dict[str, Any], roughness_score: float, metadata: Dict[str, Any]) -> List[str]:
    """Create lightweight human readable hints based on feature summary."""
    reasons: List[str] = []

    score_range = summary.get("score_range", 0.0)
    if score_range >= 15:
        reasons.append(f"実力差が大きく（{score_range:.1f}点差）、本命が堅い傾向です。")
    elif score_range <= 5:
        reasons.append(f"実力が拮抗しており（{score_range:.1f}点差）、誰が勝ってもおかしくありません。")

    diversity = summary.get("style_diversity", 0.0)
    if diversity >= 0.6:
        reasons.append("脚質が分散しており、展開が読みづらい混戦模様です。")
    
    nige_count = summary.get("style_nige_count", 0)
    if nige_count >= 4:
        reasons.append("先行選手が多く、激しい主導権争いで波乱の可能性があります。")

    if roughness_score >= 80:
        reasons.append("【激荒れ注意】過去データでも高配当が頻発するパターンです。")
    elif roughness_score <= 20:
        reasons.append("【本命党推奨】順当な決着が期待できる条件が揃っています。")

    # Remove duplicates while preserving order.
    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped[:4]


def build_betting_plan(
    roughness_score: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create heuristic betting guidance based on roughness score."""
    
    ticket_plan: List[Dict[str, str]]
    money_management: str
    hedge_note: str
    plan_summary: str
    risk_level: str

    if roughness_score >= 80:
        risk_level = "ハイリスク（激荒れ）"
        plan_summary = "本命総崩れも視野に。手広く構えて高配当を狙い撃つ局面です。"
        ticket_plan = [
            {
                "label": "三連単BOX / フォーメーション",
                "description": "有力選手だけでなく、展開利のある穴選手を含めたBOXや手広いフォーメーション推奨。",
            }
        ]
        money_management = "資金の20%程度を穴目に配分。的中率は下がるがリターンは大きい。"
        hedge_note = "ワイドで穴-穴を少額押さえるのも有効。"
    elif roughness_score >= 60:
        risk_level = "ミドルリスク（波乱含み）"
        plan_summary = "本命は信頼しきれず。ヒモ荒れや着順のズレを想定した買い目を。"
        ticket_plan = [
            {
                "label": "三連単 1頭軸マルチ",
                "description": "軸は決めても着順は決めつけないマルチ投票が安全。",
            }
        ]
        money_management = "均等買い推奨。トリガミに注意。"
        hedge_note = "本命からのワイドで保険を。"
    elif roughness_score >= 40:
        risk_level = "標準（中穴狙い）"
        plan_summary = "基本は順当だが、3着に穴が飛び込む可能性も。"
        ticket_plan = [
            {
                "label": "三連単 本命-対抗-流し",
                "description": "1-2着は固め、3着を少し広げる形がベスト。",
            }
        ]
        money_management = "本命サイドに厚く張る。"
        hedge_note = "特になし。"
    else:
        risk_level = "ローリスク（鉄板）"
        plan_summary = "極めて堅いレース。点数を絞って厚く張るか、見送りが賢明。"
        ticket_plan = [
            {
                "label": "三連単 1点〜4点",
                "description": "ガチガチの本命ラインで決まる可能性大。点数を絞る。",
            }
        ]
        money_management = "少点数に集中投資。"
        hedge_note = "なし。外れたら事故と割り切る。"

    return {
        "risk_level": risk_level,
        "plan_summary": plan_summary,
        "ticket_plan": ticket_plan,
        "money_management": money_management,
        "hedge_note": hedge_note,
        "roughness_score": roughness_score,
        "weather": summary.get("weather_condition", ""),
        "track_condition": summary.get("track_condition", ""),
    }


 

# Reworking predict_probability to call calculate_roughness_score
def predict_probability(
    feature_frame: pd.DataFrame,
    model: Any,
    metadata: Dict[str, Any],
    race_context: Dict[str, Any] = None,
) -> float:
    """
    Returns the Roughness Score (0-100).
    """
    row = feature_frame.iloc[0]
    score = calculate_roughness_score(row, metadata)
    
    # Apply track/category adjustment to the SCORE
    if race_context and TRACK_CATEGORY_STATS:
        # Simple adjustment: +/- 10% based on track multiplier
        track = race_context.get('track', '')
        track_rates = TRACK_CATEGORY_STATS.get('track_high_payout_rates', {})
        overall_rate = TRACK_CATEGORY_STATS.get('overall_high_payout_rate', 0.266)
        
        if track in track_rates:
            ratio = track_rates[track] / overall_rate
            if ratio > 1.1:
                score += 5
            elif ratio < 0.9:
                score -= 5
                
    return max(0.0, min(100.0, score))


def build_prediction_response(
    roughness_score: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    race_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    
    if roughness_score >= 80:
        confidence = "高"
        recommendation = "★激荒れ予報★ 穴党の出番です"
    elif roughness_score >= 60:
        confidence = "中"
        recommendation = "波乱含み。手広く構えましょう"
    elif roughness_score >= 40:
        confidence = "中"
        recommendation = "中穴狙い。展開次第で配当妙味あり"
    else:
        confidence = "低"
        recommendation = "本命サイド。点数を絞って厚めに"

    reasons = generate_reasons(summary, roughness_score, metadata)
    betting_plan = build_betting_plan(roughness_score, summary, metadata)
    
    betting_data = {}
    if race_info:
        try:
            betting_data = betting_suggestions.generate_tiered_suggestions(
                race_info=race_info,
                roughness_score=roughness_score,
                confidence=confidence
            )
        except Exception as e:
            print(f"Error generating betting suggestions: {e}")
            betting_data = {"error": str(e)}

    return {
        "roughness_score": roughness_score,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasons": reasons,
        "betting_plan": betting_plan,
        "betting_data": betting_data,
    }


def generate_reasons(summary: Dict[str, Any], roughness_score: float, metadata: Dict[str, Any]) -> List[str]:
    """Create lightweight human readable hints based on feature summary."""
    reasons: List[str] = []

    score_range = summary.get("score_range", 0.0)
    if score_range >= 15:
        reasons.append(f"実力差が大きく（{score_range:.1f}点差）、本命が堅い傾向です。")
    elif score_range <= 5:
        reasons.append(f"実力が拮抗しており（{score_range:.1f}点差）、誰が勝ってもおかしくありません。")

    diversity = summary.get("style_diversity", 0.0)
    if diversity >= 0.6:
        reasons.append("脚質が分散しており、展開が読みづらい混戦模様です。")
    
    nige_count = summary.get("style_nige_count", 0)
    if nige_count >= 4:
        reasons.append("先行選手が多く、激しい主導権争いで波乱の可能性があります。")

    if roughness_score >= 80:
        reasons.append("【激荒れ注意】過去データでも高配当が頻発するパターンです。")
    elif roughness_score <= 20:
        reasons.append("【本命党推奨】順当な決着が期待できる条件が揃っています。")

    # Remove duplicates while preserving order.
    seen = set()
    deduped: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)
    return deduped[:4]


def build_betting_plan(
    roughness_score: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create heuristic betting guidance based on roughness score."""
    
    ticket_plan: List[Dict[str, str]]
    money_management: str
    hedge_note: str
    plan_summary: str
    risk_level: str

    if roughness_score >= 80:
        risk_level = "ハイリスク（激荒れ）"
        plan_summary = "本命総崩れも視野に。手広く構えて高配当を狙い撃つ局面です。"
        ticket_plan = [
            {
                "label": "三連単BOX / フォーメーション",
                "description": "有力選手だけでなく、展開利のある穴選手を含めたBOXや手広いフォーメーション推奨。",
            }
        ]
        money_management = "資金の20%程度を穴目に配分。的中率は下がるがリターンは大きい。"
        hedge_note = "ワイドで穴-穴を少額押さえるのも有効。"
    elif roughness_score >= 60:
        risk_level = "ミドルリスク（波乱含み）"
        plan_summary = "本命は信頼しきれず。ヒモ荒れや着順のズレを想定した買い目を。"
        ticket_plan = [
            {
                "label": "三連単 1頭軸マルチ",
                "description": "軸は決めても着順は決めつけないマルチ投票が安全。",
            }
        ]
        money_management = "均等買い推奨。トリガミに注意。"
        hedge_note = "本命からのワイドで保険を。"
    elif roughness_score >= 40:
        risk_level = "標準（中穴狙い）"
        plan_summary = "基本は順当だが、3着に穴が飛び込む可能性も。"
        ticket_plan = [
            {
                "label": "三連単 本命-対抗-流し",
                "description": "1-2着は固め、3着を少し広げる形がベスト。",
            }
        ]
        money_management = "本命サイドに厚く張る。"
        hedge_note = "特になし。"
    else:
        risk_level = "ローリスク（鉄板）"
        plan_summary = "極めて堅いレース。点数を絞って厚く張るか、見送りが賢明。"
        ticket_plan = [
            {
                "label": "三連単 1点〜4点",
                "description": "ガチガチの本命ラインで決まる可能性大。点数を絞る。",
            }
        ]
        money_management = "少点数に集中投資。"
        hedge_note = "なし。外れたら事故と割り切る。"

    return {
        "risk_level": risk_level,
        "plan_summary": plan_summary,
        "ticket_plan": ticket_plan,
        "money_management": money_management,
        "hedge_note": hedge_note,
        "roughness_score": roughness_score,
        "weather": summary.get("weather_condition", ""),
        "track_condition": summary.get("track_condition", ""),
    }


# Final build_prediction_response - takes 4 arguments
def build_prediction_response(
    roughness_score: float,
    summary: Dict[str, Any],
    metadata: Dict[str, Any],
    race_info: Dict[str, Any] = None,
) -> Dict[str, Any]:
    
    if roughness_score >= 80:
        confidence = "高"
        recommendation = "★激荒れ予報★ 穴党の出番です"
    elif roughness_score >= 60:
        confidence = "中"
        recommendation = "波乱含み。手広く構えましょう"
    elif roughness_score >= 40:
        confidence = "中"
        recommendation = "中穴狙い。展開次第で配当妙味あり"
    else:
        confidence = "高"
        recommendation = "本命党推奨。堅い決着が濃厚"

    reasons = generate_reasons(summary, roughness_score, metadata)
    
    # Generate tiered betting suggestions
    # Use race_info if provided, otherwise fall back to summary (which may contain riders list)
    suggestions_source = race_info if race_info else summary
    betting_data = betting_suggestions.generate_tiered_suggestions(
        suggestions_source, # race_info or summary contains 'riders' list with details
        roughness_score,
        confidence
    )
    
    # Format for display
    betting_plan = betting_suggestions.format_betting_suggestions(betting_data)

    return {
        "roughness_score": roughness_score, # New field
        "probability": roughness_score / 100.0, # Legacy field for compatibility if needed
        "confidence": confidence,
        "recommendation": recommendation,
        "reasons": reasons,
        "summary": summary,
        "betting_plan": betting_plan,
        "betting_data": betting_data, # Raw data for frontend if needed
    }
