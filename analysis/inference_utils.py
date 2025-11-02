#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility helpers for real-time inference of high-payout races."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import train_high_payout_model as base_model

RIDER_NUMERIC_FIELDS = [
    "heikinTokuten",
    "nigeCnt",
    "makuriCnt",
    "sasiCnt",
    "markCnt",
    "backCnt",
]


def _safe_numeric(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_rider_stats(riders: Iterable[Dict[str, object]]) -> Dict[str, float]:
    df = pd.DataFrame(list(riders))
    if df.empty:
        raise ValueError("riders payload must contain at least one entry")

    for field in RIDER_NUMERIC_FIELDS:
        df[field] = df[field].apply(_safe_numeric)

    stats: Dict[str, float] = {}

    for prefix in [
        "heikinTokuten",
        "nigeCnt",
        "makuriCnt",
        "sasiCnt",
        "markCnt",
        "backCnt",
    ]:
        series = df[prefix]
        stats[f"{prefix}_mean"] = float(series.mean())
        stats[f"{prefix}_std"] = float(series.std(ddof=0)) if len(series) > 1 else 0.0
        stats[f"{prefix}_min"] = float(series.min())
        stats[f"{prefix}_max"] = float(series.max())

    style_counts = df.get("style") if "style" in df.columns else pd.Series(dtype=str)
    counts = style_counts.fillna("").map(base_model.STYLE_ALIAS.get).fillna(style_counts)
    for alias in base_model.STYLE_ALIAS.values():
        stats[alias] = float((counts == alias).sum())

    return stats


def build_feature_dataframe(payload: Dict[str, object]) -> pd.DataFrame:
    race_info = payload.get("race_info", {})
    riders = payload.get("riders", [])
    if not riders:
        raise ValueError("Payload must include 'riders' with at least one entry.")

    stats = _aggregate_rider_stats(riders)

    race_date = race_info.get("race_date")
    keirin_cd = str(race_info.get("keirin_cd", "")).zfill(2)
    race_no = int(race_info.get("race_no", 1))

    row: Dict[str, object] = {
        "race_date": int(race_date) if race_date else 0,
        "keirin_cd": keirin_cd,
        "race_no_int": race_no,
        "track": race_info.get("track", ""),
        "grade": race_info.get("grade", ""),
        "category": race_info.get("category", ""),
        "meeting_icon": race_info.get("meeting_icon", ""),
        "trifecta_payout": np.nan,
        "trifecta_popularity": np.nan,
        "entry_count": _safe_numeric(race_info.get("entry_count"), len(riders)),
        "narabi_flg": _safe_numeric(race_info.get("narabi_flg")),
        "narabi_y_cnt": _safe_numeric(race_info.get("narabi_y_cnt")),
        "seri": _safe_numeric(race_info.get("seri")),
        "ozz_flg": _safe_numeric(race_info.get("ozz_flg")),
        "vote_flg": _safe_numeric(race_info.get("vote_flg")),
    }
    row.update(stats)

    df = pd.DataFrame([row])
    df = base_model.add_derived_features(df)
    numeric_features, categorical_features = base_model.select_feature_columns(df)

    missing_numeric = [col for col in numeric_features if col not in df.columns]
    for col in missing_numeric:
        df[col] = 0.0
    missing_categorical = [col for col in categorical_features if col not in df.columns]
    for col in missing_categorical:
        df[col] = ""

    return df, numeric_features, categorical_features
