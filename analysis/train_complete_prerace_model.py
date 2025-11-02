#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完全版事前予測モデル - 実戦投入可能な高配当予測

利用する情報:
✓ 選手の詳細情報（平均得点、脚質、階級、出走回数）
✓ 地元選手の有無・割合
✓ 連続出走日数の推定
✓ 並び・ライン戦略
✓ 選手間の実力格差
✓ レース条件（会場、グレード、日程）
✓ 開催日の情報（初日、最終日）

除外する情報:
✗ 人気順位（結果データ）
✗ 配当金額（ターゲット変数のみ）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 基本設定
DATA_DIR = Path("data")
MODEL_DIR = Path("analysis") / "model_outputs"

# 会場コードと地域のマッピング（主要会場）
VENUE_TO_PREFECTURE = {
    "01": "北海道", "11": "千葉", "12": "川崎", "13": "東京", "14": "千葉",
    "21": "前橋", "22": "取手", "23": "宇都宮", "24": "大宮", "25": "西武",
    "27": "京王閣", "31": "松戸", "32": "名古屋", "33": "岐阜", "34": "大垣",
    "35": "富山", "36": "静岡", "41": "四日市", "42": "京都", "43": "向日町",
    "44": "奈良", "45": "岸和田", "46": "和歌山", "51": "玉野", "52": "広島",
    "53": "防府", "54": "高松", "55": "小松島", "56": "高知", "61": "松山",
    "62": "小倉", "63": "久留米", "64": "武雄", "65": "佐世保", "71": "別府",
    "72": "熊本", "73": "大分", "81": "青森", "82": "いわき平", "83": "福島",
}

# 地域グループ
VENUE_TO_REGION = {
    "01": "北海道", "11": "南関東", "12": "南関東", "13": "南関東", "14": "南関東",
    "21": "北関東", "22": "北関東", "23": "北関東", "24": "南関東", "25": "南関東",
    "27": "南関東", "31": "南関東", "32": "中部", "33": "中部", "34": "中部",
    "35": "北陸", "36": "東海", "41": "東海", "42": "近畿", "43": "近畿",
    "44": "近畿", "45": "近畿", "46": "近畿", "51": "中国", "52": "中国",
    "53": "中国", "54": "四国", "55": "四国", "56": "四国", "61": "四国",
    "62": "九州", "63": "九州", "64": "九州", "65": "九州", "71": "九州",
    "72": "九州", "73": "九州", "81": "東北", "82": "東北", "83": "東北",
}

# 府県から地域へのマッピング
PREFECTURE_TO_REGION = {
    "北海道": "北海道",
    "青森": "東北", "岩手": "東北", "宮城": "東北", "秋田": "東北", "山形": "東北", "福島": "東北",
    "茨城": "北関東", "栃木": "北関東", "群馬": "北関東",
    "埼玉": "南関東", "千葉": "南関東", "東京": "南関東", "神奈川": "南関東",
    "新潟": "北陸", "富山": "北陸", "石川": "北陸", "福井": "北陸",
    "山梨": "中部", "長野": "中部", "岐阜": "中部", "静岡": "東海", "愛知": "東海",
    "三重": "東海", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿", "兵庫": "近畿",
    "奈良": "近畿", "和歌山": "近畿",
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州", "大分": "九州",
    "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


def load_results_for_training(path: Path, payout_threshold: int) -> pd.DataFrame:
    """結果データ読み込み（人気順位は使用しない）"""
    usecols = [
        "race_date", "keirin_cd", "race_no", "track", "grade",
        "category", "meeting_icon", "trifecta_payout",
    ]
    df = pd.read_csv(path, usecols=usecols)

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
        .fillna(0)
        .astype(int)
    )

    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)
    df["race_date"] = df["race_date"].astype(int)

    return df


def load_prerace_with_lineup(path: Path) -> pd.DataFrame:
    """
    レース前情報を読み込み、並び・ライン情報を抽出

    並び情報:
    - narabi_flg: 並びフラグ
    - narabi_y_cnt: 並び予想数
    - entry1~entry9_assen: 各選手の並び予想（印）
    """
    df = pd.read_csv(path, dtype={"bkeirin_cd": str, "keirin_cd": str})

    df["race_date"] = df["race_date"].astype(int)
    df["race_no"] = df["race_no"].astype(int)
    df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)

    # 日付特徴量
    df["race_date_str"] = df["race_date"].astype(str)
    df["year"] = df["race_date_str"].str[:4].astype(int)
    df["month"] = df["race_date_str"].str[4:6].astype(int)
    df["day"] = df["race_date_str"].str[6:8].astype(int)

    df["date_obj"] = pd.to_datetime(df["race_date_str"], format="%Y%m%d", errors="coerce")
    df["day_of_week"] = df["date_obj"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)

    # 開催日程（nitiji列から）
    if "nitiji" in df.columns:
        df["is_first_day"] = df["nitiji"].str.contains("初日", na=False).astype(int)
        df["is_final_day"] = df["nitiji"].str.contains("最終", na=False).astype(int)
        df["is_second_day"] = df["nitiji"].str.contains("2日目|二日目", na=False).astype(int)

    # グレード特徴量
    if "grade" in df.columns:
        df["is_gp"] = df["grade"].str.contains("GP", na=False).astype(int)
        df["is_g1"] = df["grade"].str.contains("G1", na=False).astype(int)
        df["is_g2"] = df["grade"].str.contains("G2", na=False).astype(int)
        df["is_g3"] = df["grade"].str.contains("G3", na=False).astype(int)
        df["is_f1"] = df["grade"].str.contains("F1", na=False).astype(int)
        df["is_f2"] = df["grade"].str.contains("F2", na=False).astype(int)

    # 並び情報の特徴量化
    if "narabi_flg" in df.columns:
        df["has_narabi"] = df["narabi_flg"].fillna(0).astype(int)

    if "narabi_y_cnt" in df.columns:
        df["narabi_count"] = pd.to_numeric(df["narabi_y_cnt"], errors="coerce").fillna(0)

    # 並び予想の複雑さ（印の種類）
    assen_cols = [f"entry{i}_assen" for i in range(1, 10) if f"entry{i}_assen" in df.columns]
    if len(assen_cols) > 0:
        # 印の数をカウント
        df["assen_total_count"] = df[assen_cols].notna().sum(axis=1)

        # 追加印（color_blue）の数
        for col in assen_cols:
            if col in df.columns:
                df[f"{col}_is_追加"] = df[col].str.contains("追加", na=False).astype(int)

        追加_cols = [f"{col}_is_追加" for col in assen_cols if f"{col}_is_追加" in df.columns]
        if len(追加_cols) > 0:
            df["assen_追加_count"] = df[追加_cols].sum(axis=1)

    return df


def load_rider_details(path: Path) -> pd.DataFrame:
    """
    選手の詳細情報を読み込み

    各選手ごとに:
    - 登録番号（sensyuRegistNo）
    - 名前（sensyuName）
    - 府県（huKen）
    - 階級（kyuhan）
    - 脚質（kyakusitu）
    - 平均得点（heikinTokuten）
    - 出走回数（nigeCnt, makuriCnt, etc）
    - 車番（syaban）
    """
    usecols = [
        "race_encp", "race_date", "track", "keirin_cd",
        "syaban", "sensyuRegistNo", "sensyuName",
        "huKen", "prevKyuhan", "kyuhan", "kyakusitu",
        "heikinTokuten", "nigeCnt", "makuriCnt",
        "sasiCnt", "markCnt", "backCnt",
    ]

    # 列の存在確認
    df_peek = pd.read_csv(path, nrows=1)
    usecols = [col for col in usecols if col in df_peek.columns]

    df = pd.read_csv(path, usecols=usecols)

    # 数値変換
    numeric_cols = ["heikinTokuten", "nigeCnt", "makuriCnt", "sasiCnt", "markCnt", "backCnt"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["race_date"] = df["race_date"].astype(int)

    if "keirin_cd" in df.columns:
        df["keirin_cd"] = df["keirin_cd"].astype(str).str.zfill(2)

    return df


def calculate_consecutive_races(rider_df: pd.DataFrame) -> pd.DataFrame:
    """
    選手の連続出走日数を計算

    ロジック:
    1. 選手IDと日付でソート
    2. 同じ選手の連続する日付の差分を計算
    3. 1日なら連続、2日以上なら非連続
    """
    if "sensyuRegistNo" not in rider_df.columns or "race_date" not in rider_df.columns:
        return rider_df

    # 選手IDと日付でソート
    rider_df = rider_df.sort_values(["sensyuRegistNo", "race_date"])

    # 同じ選手の前回出走日を計算
    rider_df["prev_race_date"] = rider_df.groupby("sensyuRegistNo")["race_date"].shift(1)

    # 日数差分を計算
    rider_df["days_since_last_race"] = (
        pd.to_datetime(rider_df["race_date"].astype(str), format="%Y%m%d")
        - pd.to_datetime(rider_df["prev_race_date"].astype(str), format="%Y%m%d", errors="coerce")
    ).dt.days

    # 連続出走フラグ（前日も出走していた場合）
    rider_df["is_consecutive_race"] = (rider_df["days_since_last_race"] == 1).astype(int)

    # 連続出走日数（簡易版：前日出走なら2日目、そうでなければ1日目）
    rider_df["consecutive_days"] = rider_df["is_consecutive_race"] + 1

    return rider_df


def aggregate_riders_per_race(rider_df: pd.DataFrame) -> pd.DataFrame:
    """
    レースごとに選手情報を集約

    追加する特徴量:
    - 地元選手の数・割合
    - 連続出走選手の数・割合
    - 選手の実力統計（平均得点の平均、標準偏差、範囲）
    - 脚質の分布
    - 階級の分布
    - 選手間の相性指標
    """

    # 会場コードから地域を取得
    rider_df["venue_region"] = rider_df["keirin_cd"].map(VENUE_TO_REGION)

    # 選手の府県から地域を取得
    if "huKen" in rider_df.columns:
        # 府県名の正規化（全角スペース削除、「　」削除）
        rider_df["huKen_clean"] = (
            rider_df["huKen"]
            .astype(str)
            .str.replace("　", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.strip()
        )
        rider_df["rider_region"] = rider_df["huKen_clean"].map(PREFECTURE_TO_REGION)

        # 地元選手フラグ（同じ地域）
        rider_df["is_local"] = (
            rider_df["venue_region"] == rider_df["rider_region"]
        ).astype(int)
    else:
        rider_df["is_local"] = 0

    # レースごとにグループ化
    grouped = rider_df.groupby("race_encp")

    # 基本統計量
    agg_dict = {
        "syaban": "count",  # 選手数
    }

    numeric_cols = ["heikinTokuten", "nigeCnt", "makuriCnt", "sasiCnt", "markCnt", "backCnt"]
    for col in numeric_cols:
        if col in rider_df.columns:
            agg_dict[col] = ["mean", "std", "min", "max"]

    # 連続出走と地元選手
    if "is_consecutive_race" in rider_df.columns:
        agg_dict["is_consecutive_race"] = ["sum", "mean"]

    if "consecutive_days" in rider_df.columns:
        agg_dict["consecutive_days"] = ["mean", "max"]

    if "is_local" in rider_df.columns:
        agg_dict["is_local"] = ["sum", "mean"]

    agg_result = grouped.agg(agg_dict)
    agg_result.columns = [f"{col}_{stat}" if stat else col for col, stat in agg_result.columns]

    # 脚質の分布
    if "kyakusitu" in rider_df.columns:
        style_counts = rider_df.pivot_table(
            index="race_encp",
            columns="kyakusitu",
            values="syaban",
            aggfunc="count",
            fill_value=0,
        )

        # 列名を標準化
        style_mapping = {"逃": "style_nige", "追": "style_oikomi", "両": "style_ryo"}
        style_counts = style_counts.rename(columns=lambda x: style_mapping.get(x, f"style_{x}"))

        agg_result = agg_result.join(style_counts, how="left")

        # 脚質の多様性（エントロピー）
        style_cols = [col for col in agg_result.columns if col.startswith("style_")]
        if len(style_cols) >= 2:
            style_total = agg_result[style_cols].sum(axis=1)
            agg_result["style_diversity"] = 0
            for col in style_cols:
                p = agg_result[col] / (style_total + 1e-6)
                agg_result["style_diversity"] -= p * np.log(p + 1e-6)

    # 階級の分布
    if "kyuhan" in rider_df.columns:
        grade_counts = rider_df.pivot_table(
            index="race_encp",
            columns="kyuhan",
            values="syaban",
            aggfunc="count",
            fill_value=0,
        )

        # S級、A級の数
        for grade in ["S1", "S2", "A1", "A2", "A3"]:
            col_name = f"grade_{grade}_count"
            if grade in grade_counts.columns:
                agg_result[col_name] = grade_counts[grade]
            else:
                agg_result[col_name] = 0

    # 実力格差の追加指標
    if "heikinTokuten_max" in agg_result.columns and "heikinTokuten_min" in agg_result.columns:
        agg_result["tokuten_range"] = (
            agg_result["heikinTokuten_max"] - agg_result["heikinTokuten_min"]
        )

        # 変動係数（CV）
        agg_result["tokuten_cv"] = (
            agg_result["heikinTokuten_std"] / (agg_result["heikinTokuten_mean"].abs() + 0.01)
        )

    # 出走経験の統計
    exp_cols = ["nigeCnt_mean", "makuriCnt_mean", "sasiCnt_mean", "markCnt_mean", "backCnt_mean"]
    existing_exp_cols = [col for col in exp_cols if col in agg_result.columns]
    if len(existing_exp_cols) >= 2:
        agg_result["total_race_experience"] = agg_result[existing_exp_cols].sum(axis=1)

    return agg_result.reset_index()


def merge_all_datasets(
    results: pd.DataFrame,
    prerace: pd.DataFrame,
    rider_agg: pd.DataFrame,
) -> pd.DataFrame:
    """すべてのデータセットを統合"""

    # レース前情報と選手集約データを結合
    prerace_enriched = prerace.merge(rider_agg, on="race_encp", how="left")

    # 結果データと結合
    merged = results.merge(
        prerace_enriched,
        left_on=["race_date", "keirin_cd", "race_no_int"],
        right_on=["race_date", "keirin_cd", "race_no"],
        how="left",
        suffixes=("", "_pre"),
    )

    # 不要列削除
    cols_to_drop = [col for col in merged.columns if col.endswith("_pre")]
    if "race_no" in merged.columns and "race_no_int" in merged.columns:
        cols_to_drop.append("race_no")

    merged = merged.drop(columns=cols_to_drop, errors="ignore")

    return merged


def select_all_prerace_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """事前予測用の全特徴量を選択"""

    # カテゴリ特徴量
    categorical_features = []
    for col in ["track", "grade", "category", "meeting_icon", "keirin_cd", "syumoku"]:
        if col in df.columns:
            categorical_features.append(col)

    # 数値特徴量
    numeric_features = []

    # パターンマッチ
    stat_patterns = [
        "heikinTokuten_", "nigeCnt_", "makuriCnt_", "sasiCnt_",
        "markCnt_", "backCnt_", "tokuten_", "style_", "grade_",
    ]

    for col in df.columns:
        # 統計量
        if any(col.startswith(pat) for pat in stat_patterns):
            if df[col].dtype in ["int64", "float64"]:
                numeric_features.append(col)

        # レース情報
        elif col in [
            "race_no_int", "entry_count", "narabi_flg", "narabi_count",
            "has_narabi", "assen_total_count", "assen_追加_count",
            "year", "month", "day", "day_of_week",
            "is_weekend", "is_monday", "is_first_day", "is_second_day", "is_final_day",
            "is_gp", "is_g1", "is_g2", "is_g3", "is_f1", "is_f2",
            "syaban_count", "style_diversity", "total_race_experience",
            "is_consecutive_race_sum", "is_consecutive_race_mean",
            "consecutive_days_mean", "consecutive_days_max",
            "is_local_sum", "is_local_mean",
        ]:
            if col in df.columns and df[col].dtype in ["int64", "float64"]:
                numeric_features.append(col)

    # 除外確認
    excluded = ["trifecta_popularity", "trifecta_payout", "target_high_payout"]
    numeric_features = [f for f in numeric_features if f not in excluded]
    categorical_features = [f for f in categorical_features if f not in excluded]

    # 重複削除
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    return numeric_features, categorical_features


def build_complete_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], List[str]]:
    """完全版データセットを構築"""

    print("=" * 70)
    print("完全版事前予測データセット構築")
    print("=" * 70)

    print("\n[1/6] Loading results data...")
    results = load_results_for_training(results_path, payout_threshold)
    print(f"  ✓ Results: {len(results):,} races")

    print("\n[2/6] Loading prerace data with lineup info...")
    prerace = load_prerace_with_lineup(prerace_path)
    print(f"  ✓ Prerace: {len(prerace):,} races")

    print("\n[3/6] Loading rider details...")
    rider_df = load_rider_details(entries_path)
    print(f"  ✓ Riders: {len(rider_df):,} entries")

    print("\n[4/6] Calculating consecutive races...")
    rider_df = calculate_consecutive_races(rider_df)
    consecutive_count = rider_df["is_consecutive_race"].sum()
    print(f"  ✓ Consecutive races detected: {consecutive_count:,} entries")

    print("\n[5/6] Aggregating rider features per race...")
    rider_agg = aggregate_riders_per_race(rider_df)
    print(f"  ✓ Aggregated: {len(rider_agg):,} races")

    print("\n[6/6] Merging all datasets...")
    dataset = merge_all_datasets(results, prerace, rider_agg)
    print(f"  ✓ Merged: {len(dataset):,} races")

    print("\n[7/7] Selecting features...")
    numeric_features, categorical_features = select_all_prerace_features(dataset)

    # 時系列ソート
    dataset = dataset.sort_values(["race_date", "keirin_cd", "race_no_int"]).reset_index(drop=True)

    # 特徴量マトリックス
    feature_columns = numeric_features + categorical_features

    # 欠損確認
    missing_cols = [col for col in feature_columns if col not in dataset.columns]
    if missing_cols:
        print(f"  ⚠ Missing columns: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in dataset.columns]
        numeric_features = [f for f in numeric_features if f in dataset.columns]
        categorical_features = [f for f in categorical_features if f in dataset.columns]

    X = dataset[feature_columns].copy()

    for col in categorical_features:
        X[col] = X[col].astype(str).astype("category")

    y = dataset["target_high_payout"].astype(int)

    print("\n" + "=" * 70)
    print("データセット概要")
    print("=" * 70)
    print(f"総サンプル数:       {len(dataset):,}")
    print(f"高配当レース数:     {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"数値特徴量:         {len(numeric_features)}")
    print(f"カテゴリ特徴量:     {len(categorical_features)}")
    print(f"総特徴量数:         {len(feature_columns)}")
    print(f"\n✓ 地元選手情報:     含む")
    print(f"✓ 連続出走情報:     含む")
    print(f"✓ 並び・ライン情報: 含む")
    print(f"✓ 選手実力格差:     含む")
    print(f"✗ 人気順位:         除外（事前予測不可）")
    print("=" * 70)

    return dataset, X, y, numeric_features, categorical_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="完全版事前予測モデルのデータセット構築",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--results", default=DATA_DIR / "keirin_results_20240101_20251004.csv", type=Path)
    parser.add_argument("--prerace", default=DATA_DIR / "keirin_prerace_20240101_20251004.csv", type=Path)
    parser.add_argument("--entries", default=DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv", type=Path)
    parser.add_argument("--threshold", default=10000, type=int)
    parser.add_argument("--output-dataset", default=MODEL_DIR / "complete_prerace_dataset.csv", type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # データセット構築
    dataset, X, y, numeric_features, categorical_features = build_complete_dataset(
        args.results,
        args.prerace,
        args.entries,
        args.threshold,
    )

    # 保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    full_dataset = X.copy()
    full_dataset["target_high_payout"] = y
    full_dataset["race_date"] = dataset["race_date"]
    full_dataset["keirin_cd"] = dataset["keirin_cd"]
    full_dataset["race_no"] = dataset["race_no_int"]
    full_dataset["trifecta_payout"] = dataset["trifecta_payout"]

    full_dataset.to_csv(args.output_dataset, index=False)
    print(f"\n✓ Dataset saved: {args.output_dataset}")

    # 特徴量情報
    feature_info = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "total_features": len(numeric_features) + len(categorical_features),
        "features_included": {
            "rider_stats": True,
            "local_riders": True,
            "consecutive_races": True,
            "lineup_strategy": True,
            "skill_gap": True,
            "race_conditions": True,
        },
        "features_excluded": {
            "popularity_rank": True,
            "payout_amount": True,
        },
        "note": "このモデルは事前予測可能な情報のみを使用しています",
    }

    feature_info_path = MODEL_DIR / "complete_prerace_feature_info.json"
    with open(feature_info_path, "w", encoding="utf-8") as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    print(f"✓ Feature info saved: {feature_info_path}")

    # サンプル特徴量を表示
    print("\n" + "=" * 70)
    print("主要な特徴量サンプル")
    print("=" * 70)

    feature_categories = {
        "地元選手": [f for f in numeric_features if "local" in f],
        "連続出走": [f for f in numeric_features if "consecutive" in f],
        "並び戦略": [f for f in numeric_features if "narabi" in f or "assen" in f],
        "実力統計": [f for f in numeric_features if "tokuten" in f][:5],
        "脚質分布": [f for f in numeric_features if "style" in f][:5],
    }

    for category, features in feature_categories.items():
        if features:
            print(f"\n{category}:")
            for feat in features[:3]:  # 最大3つ表示
                print(f"  - {feat}")

    print("\n" + "=" * 70)
    print("次のステップ:")
    print("  1. LightGBMで訓練")
    print("  2. TimeSeriesSplitで交差検証")
    print("  3. 特徴量重要度を確認")
    print("=" * 70)


if __name__ == "__main__":
    main()
