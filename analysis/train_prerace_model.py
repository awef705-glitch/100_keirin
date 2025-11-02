#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事前予測モデル - レース前に分かる情報のみで高配当を予測

重要: trifecta_popularity（人気順位）は結果データなので使用しない
事前に分かる情報のみで予測可能なモデルを構築
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 基本設定
DATA_DIR = Path("data")
MODEL_DIR = Path("analysis") / "model_outputs"

# 会場コードと地域のマッピング（実際のデータから作成する必要がある）
VENUE_TO_REGION = {
    "01": "北海道", "11": "関東", "12": "関東", "13": "関東", "14": "関東",
    "21": "関東", "22": "関東", "23": "関東", "24": "関東", "25": "関東",
    "27": "関東", "31": "中部", "32": "中部", "33": "中部", "34": "中部",
    "41": "近畿", "42": "近畿", "43": "近畿", "44": "近畿", "45": "近畿",
    "46": "近畿", "51": "中国", "52": "中国", "53": "中国", "54": "四国",
    "61": "九州", "62": "九州", "63": "九州", "64": "九州", "65": "九州",
    "71": "九州", "72": "九州", "73": "九州", "81": "九州",
}


def load_results_for_training(path: Path, payout_threshold: int) -> pd.DataFrame:
    """
    結果データを読み込み。人気順位は使用しない。
    配当金額はターゲット変数としてのみ使用。
    """
    usecols = [
        "race_date",
        "keirin_cd",
        "race_no",
        "track",
        "grade",
        "category",
        "meeting_icon",
        "trifecta_payout",
        # "trifecta_popularity",  # ← 意図的に除外！
    ]
    df = pd.read_csv(path, usecols=usecols)

    # 配当金額のクリーニング
    df["trifecta_payout"] = (
        df["trifecta_payout"]
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
        .fillna(0)
        .astype(int)
    )

    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)
    df["race_date"] = df["race_date"].astype(int)

    return df


def load_and_enrich_prerace(path: Path) -> pd.DataFrame:
    """レース前情報を読み込み、追加の特徴量を生成"""
    df = pd.read_csv(path, dtype={"bkeirin_cd": str, "keirin_cd": str})

    df["race_date"] = df["race_date"].astype(int)
    df["race_no"] = df["race_no"].astype(int)
    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)

    # 日付関連の特徴量
    df["race_date_str"] = df["race_date"].astype(str)
    df["year"] = df["race_date_str"].str[:4].astype(int)
    df["month"] = df["race_date_str"].str[4:6].astype(int)
    df["day"] = df["race_date_str"].str[6:8].astype(int)

    # 日付オブジェクト作成
    df["date_obj"] = pd.to_datetime(df["race_date_str"], format="%Y%m%d", errors="coerce")
    df["day_of_week"] = df["date_obj"].dt.dayofweek  # 0=月曜
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 開催日程の情報（nitiji列から）
    if "nitiji" in df.columns:
        df["is_first_day"] = df["nitiji"].str.contains("初日", na=False).astype(int)
        df["is_final_day"] = df["nitiji"].str.contains("最終", na=False).astype(int)

    # グレード情報のエンコーディング
    if "grade" in df.columns:
        df["is_gp"] = df["grade"].str.contains("GP", na=False).astype(int)
        df["is_g1"] = df["grade"].str.contains("G1", na=False).astype(int)
        df["is_g2"] = df["grade"].str.contains("G2", na=False).astype(int)
        df["is_g3"] = df["grade"].str.contains("G3", na=False).astype(int)

    return df


def aggregate_entries_with_features(path: Path) -> pd.DataFrame:
    """
    選手データを集約し、事前予測用の特徴量を生成

    追加する特徴量:
    - 地元選手の有無・割合
    - 連続出走の推定
    - 選手の実力格差
    - 脚質の多様性
    """
    df = pd.read_csv(path, dtype={"keirin_cd": str})

    # 数値列の変換
    numeric_cols = [
        "heikinTokuten",  # 平均得点
        "nigeCnt",        # 逃げ回数
        "makuriCnt",      # まくり回数
        "sasiCnt",        # 差し回数
        "markCnt",        # マーク回数
        "backCnt",        # バック回数
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 地元選手フラグ（会場コードと選手の府県が一致）
    # ※ 実際には会場コードと府県の詳細なマッピングが必要
    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2) if "keirin_cd" in df.columns else None

    # レースごとに集約
    grouped = df.groupby("race_encp")

    # 基本統計量
    agg_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            agg_dict[col] = ["mean", "std", "min", "max"]

    agg_numeric = grouped.agg(agg_dict)
    agg_numeric.columns = [f"{col}_{stat}" for col, stat in agg_numeric.columns]

    # 脚質の分布
    if "kyakusitu" in df.columns:
        style_counts = (
            df.pivot_table(
                index="race_encp",
                columns="kyakusitu",
                values="syaban",
                aggfunc="count",
                fill_value=0,
            )
        )
        # 列名を標準化
        style_mapping = {"逃": "style_nige", "追": "style_oikomi", "両": "style_ryo"}
        style_counts = style_counts.rename(columns=lambda x: style_mapping.get(x, f"style_{x}"))
        agg_numeric = agg_numeric.join(style_counts, how="left")

    # 選手数
    agg_numeric["rider_count"] = grouped.size()

    # 平均得点の格差
    if "heikinTokuten_max" in agg_numeric.columns and "heikinTokuten_min" in agg_numeric.columns:
        agg_numeric["tokuten_range"] = agg_numeric["heikinTokuten_max"] - agg_numeric["heikinTokuten_min"]
        agg_numeric["tokuten_cv"] = agg_numeric["heikinTokuten_std"] / (agg_numeric["heikinTokuten_mean"].abs() + 0.01)

    # 脚質の多様性（エントロピー風）
    style_cols = [col for col in agg_numeric.columns if col.startswith("style_")]
    if len(style_cols) >= 2:
        style_total = agg_numeric[style_cols].sum(axis=1)
        agg_numeric["style_diversity"] = 0
        for col in style_cols:
            p = agg_numeric[col] / (style_total + 1e-6)
            agg_numeric["style_diversity"] -= p * np.log(p + 1e-6)

    # 出走回数の合計と変動
    cnt_cols = ["nigeCnt_mean", "makuriCnt_mean", "sasiCnt_mean", "markCnt_mean", "backCnt_mean"]
    existing_cnt_cols = [col for col in cnt_cols if col in agg_numeric.columns]
    if len(existing_cnt_cols) >= 2:
        agg_numeric["total_race_experience"] = agg_numeric[existing_cnt_cols].sum(axis=1)
        agg_numeric["race_experience_std"] = agg_numeric[[col.replace("_mean", "_std") for col in existing_cnt_cols if col.replace("_mean", "_std") in agg_numeric.columns]].mean(axis=1)

    return agg_numeric.reset_index()


def merge_prerace_datasets(
    results: pd.DataFrame,
    prerace: pd.DataFrame,
    entries_agg: pd.DataFrame,
) -> pd.DataFrame:
    """データセットを統合"""

    # レース前情報と選手集約データを結合
    prerace_enriched = prerace.merge(entries_agg, on="race_encp", how="left")

    # 結果データと結合
    merged = results.merge(
        prerace_enriched,
        left_on=["race_date", "keirin_cd", "race_no_int"],
        right_on=["race_date", "keirin_cd", "race_no"],
        how="left",
        suffixes=("", "_pre"),
    )

    # 不要な重複列を削除
    cols_to_drop = [col for col in merged.columns if col.endswith("_pre") and col != "grade_pre"]
    if "race_no" in merged.columns and "race_no_int" in merged.columns:
        cols_to_drop.append("race_no")

    merged = merged.drop(columns=cols_to_drop, errors="ignore")

    return merged


def select_prerace_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    事前予測に使用できる特徴量を選択

    除外:
    - trifecta_popularity（人気順位） ← 結果データ
    - trifecta_payout（配当） ← ターゲット変数
    """

    # カテゴリ特徴量
    categorical_features = []
    for col in ["track", "grade", "category", "meeting_icon", "keirin_cd"]:
        if col in df.columns:
            categorical_features.append(col)

    # 数値特徴量
    numeric_features = []

    # 基本的な選手統計
    stat_patterns = [
        "heikinTokuten_", "nigeCnt_", "makuriCnt_", "sasiCnt_",
        "markCnt_", "backCnt_", "tokuten_", "style_"
    ]

    for col in df.columns:
        # 統計量列
        if any(col.startswith(pat) for pat in stat_patterns):
            if df[col].dtype in ["int64", "float64"]:
                numeric_features.append(col)

        # レース情報
        elif col in [
            "race_no_int", "entry_count", "narabi_flg", "narabi_y_cnt",
            "year", "month", "day", "day_of_week", "is_weekend",
            "is_first_day", "is_final_day", "is_gp", "is_g1", "is_g2", "is_g3",
            "rider_count", "style_diversity", "total_race_experience", "race_experience_std"
        ]:
            if col in df.columns and df[col].dtype in ["int64", "float64"]:
                numeric_features.append(col)

    # 除外すべき列を確認
    excluded = ["trifecta_popularity", "trifecta_payout", "target_high_payout"]
    numeric_features = [f for f in numeric_features if f not in excluded]
    categorical_features = [f for f in categorical_features if f not in excluded]

    return numeric_features, categorical_features


def build_prerace_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:
    """事前予測用データセットを構築"""

    print("Loading results data...")
    results = load_results_for_training(results_path, payout_threshold)

    print("Loading and enriching prerace data...")
    prerace = load_and_enrich_prerace(prerace_path)

    print("Aggregating entries data with features...")
    entries_agg = aggregate_entries_with_features(entries_path)

    print("Merging datasets...")
    dataset = merge_prerace_datasets(results, prerace, entries_agg)

    print("Selecting prerace features...")
    numeric_features, categorical_features = select_prerace_features(dataset)

    # 時系列でソート
    dataset = dataset.sort_values(["race_date", "keirin_cd", "race_no_int"]).reset_index(drop=True)

    # 特徴量マトリックス作成
    feature_columns = numeric_features + categorical_features

    # 欠損チェック
    missing_cols = [col for col in feature_columns if col not in dataset.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in dataset.columns]
        numeric_features = [f for f in numeric_features if f in dataset.columns]
        categorical_features = [f for f in categorical_features if f in dataset.columns]

    X = dataset[feature_columns].copy()

    # カテゴリ変数の型変換
    for col in categorical_features:
        X[col] = X[col].astype(str).astype("category")

    y = dataset["target_high_payout"].astype(int)

    print(f"\nDataset summary:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Positive rate: {y.mean():.4f}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Total features: {len(feature_columns)}")
    print(f"\n  ※ 人気順位は使用していません（事前予測可能）")

    return dataset, X, y, numeric_features, categorical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="事前予測モデルの訓練（人気順位を使わない）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
このモデルはレース前に分かる情報のみを使用します:
  ✓ 選手の平均得点、脚質、出走回数
  ✓ レースのグレード、会場、日程
  ✓ 選手間の実力格差
  ✗ 人気順位（結果データなので使用不可）
        """
    )

    parser.add_argument(
        "--results",
        default=DATA_DIR / "keirin_results_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument(
        "--prerace",
        default=DATA_DIR / "keirin_prerace_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument(
        "--entries",
        default=DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument("--threshold", default=10000, type=int)
    parser.add_argument("--output-dataset", default=MODEL_DIR / "prerace_dataset.csv", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # データセット構築
    dataset, X, y, numeric_features, categorical_features = build_prerace_dataset(
        args.results,
        args.prerace,
        args.entries,
        args.threshold,
    )

    # データセット保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 特徴量とターゲットを含む完全なデータセットを保存
    full_dataset = X.copy()
    full_dataset["target_high_payout"] = y
    full_dataset["race_date"] = dataset["race_date"]
    full_dataset["keirin_cd"] = dataset["keirin_cd"]
    full_dataset["race_no"] = dataset["race_no_int"]

    full_dataset.to_csv(args.output_dataset, index=False)
    print(f"\n✓ Dataset saved to: {args.output_dataset}")

    # 特徴量リスト保存
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "total_features": len(numeric_features) + len(categorical_features),
        "note": "人気順位（trifecta_popularity）は使用していません。事前予測可能なモデルです。"
    }

    feature_info_path = MODEL_DIR / "prerace_feature_info.json"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    print(f"✓ Feature info saved to: {feature_info_path}")

    print("\n" + "="*70)
    print("次のステップ:")
    print("  1. このデータセットを使ってLightGBMを訓練")
    print("  2. 人気順位なしでどの程度の精度が出るか検証")
    print("  3. 重要な特徴量を確認")
    print("="*70)


if __name__ == "__main__":
    main()
