#!/usr/bin/env python3
"""
競輪予測モデルの学習と保存
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


def parse_payout(value: str) -> float:
    """配当金文字列を数値に変換"""
    if pd.isna(value) or value == "":
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def build_dataset(csv_path: Path) -> tuple:
    """データセットを構築"""
    df = pd.read_csv(csv_path)
    df["trifecta_payout_value"] = df["trifecta_payout"].apply(parse_payout)
    df = df.dropna(subset=["trifecta_payout_value"])
    df["high_payout"] = (df["trifecta_payout_value"] >= 10000).astype(int)

    # 車番の統計量
    for pos in (1, 2, 3):
        df[f"pos{pos}_car_no"] = pd.to_numeric(df[f"pos{pos}_car_no"], errors="coerce")
    df["car_sum"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].sum(axis=1)
    df["car_std"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].std(axis=1)
    df["car_range"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].max(axis=1) - df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].min(axis=1)
    df["car_median"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].median(axis=1)
    df["car_min"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].min(axis=1)
    df["car_max"] = df[["pos1_car_no", "pos2_car_no", "pos3_car_no"]].max(axis=1)

    # カテゴリカル列の処理
    cat_cols = [
        "grade",
        "track",
        "category",
        "race_no",
        "meeting_icon",
        "pos1_decision",
        "pos2_decision",
        "pos3_decision",
        "pos1_name",
        "pos2_name",
        "pos3_name",
    ]
    for col in cat_cols:
        df[col] = df[col].fillna("(欠損)").astype(str)

    df["race_no"] = df["race_no"].str.upper().str.replace("R", "", regex=False)

    # 数値列の標準化パラメータを保存
    numeric_cols = ["car_sum", "car_std", "car_range", "car_median", "car_min", "car_max"]
    standardization_params = {}
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        standardization_params[col] = {"mean": mean, "std": std}
        if std == 0:
            df[col] = 0
        else:
            df[col] = (df[col] - mean) / std

    X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=False)
    X_num = df[numeric_cols]
    X = pd.concat([X_cat, X_num], axis=1)
    y = df["high_payout"].values

    return X, y, df, standardization_params, cat_cols


def train_model(X: pd.DataFrame, y: np.ndarray):
    """モデルを学習"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "positive_rate": float(y.mean()),
    }
    return model, metrics


def main():
    # データの読み込み
    csv_path = Path("data/keirin_results_20240101_20251004.csv")
    if not csv_path.exists():
        raise SystemExit(f"データファイルが見つかりません: {csv_path}")

    print("データセットを構築中...")
    X, y, df, standardization_params, cat_cols = build_dataset(csv_path)

    print(f"総レース数: {len(df)}")
    print(f"高配当レース数: {df['high_payout'].sum()} ({df['high_payout'].mean()*100:.1f}%)")

    print("\nモデルを学習中...")
    model, metrics = train_model(X, y)

    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")

    # モデルとパラメータを保存
    model_dir = Path("backend/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # モデルの保存
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 特徴量名の保存
    feature_names = list(X.columns)
    with open(model_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # 標準化パラメータの保存
    with open(model_dir / "standardization_params.json", "w", encoding="utf-8") as f:
        json.dump(standardization_params, f, ensure_ascii=False, indent=2)

    # カテゴリカル列の保存
    with open(model_dir / "categorical_columns.json", "w", encoding="utf-8") as f:
        json.dump(cat_cols, f, ensure_ascii=False, indent=2)

    # メトリクスの保存
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 訓練データから取得可能な値のリストを保存（フロントエンド用）
    track_list = sorted(df["track"].unique().tolist())
    grade_list = sorted(df["grade"].unique().tolist())
    category_list = sorted(df["category"].unique().tolist())

    reference_data = {
        "tracks": track_list,
        "grades": grade_list,
        "categories": category_list,
    }

    with open(model_dir / "reference_data.json", "w", encoding="utf-8") as f:
        json.dump(reference_data, f, ensure_ascii=False, indent=2)

    print(f"\nモデルを保存しました: {model_dir}")
    print("- model.pkl")
    print("- feature_names.json")
    print("- standardization_params.json")
    print("- categorical_columns.json")
    print("- metrics.json")
    print("- reference_data.json")


if __name__ == "__main__":
    main()
