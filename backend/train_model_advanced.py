#!/usr/bin/env python3
"""
競輪予測モデルの学習と保存（高精度版）
LightGBM + 高度な特徴量エンジニアリング
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def parse_payout(value: str) -> float:
    """配当金文字列を数値に変換"""
    if pd.isna(value) or value == "":
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """高度な特徴量を生成"""

    # 車番の基本統計量
    for pos in (1, 2, 3):
        df[f"pos{pos}_car_no"] = pd.to_numeric(df[f"pos{pos}_car_no"], errors="coerce")

    car_cols = ["pos1_car_no", "pos2_car_no", "pos3_car_no"]

    # 基本統計量
    df["car_sum"] = df[car_cols].sum(axis=1)
    df["car_std"] = df[car_cols].std(axis=1)
    df["car_range"] = df[car_cols].max(axis=1) - df[car_cols].min(axis=1)
    df["car_median"] = df[car_cols].median(axis=1)
    df["car_min"] = df[car_cols].min(axis=1)
    df["car_max"] = df[car_cols].max(axis=1)
    df["car_mean"] = df[car_cols].mean(axis=1)

    # 高度な特徴量
    # 1. 車番の連続性（1-2-3のような連番かどうか）
    df["is_consecutive"] = (
        ((df["pos2_car_no"] - df["pos1_car_no"]).abs() == 1) &
        ((df["pos3_car_no"] - df["pos2_car_no"]).abs() == 1)
    ).astype(int)

    # 2. 車番の偶奇パターン
    df["odd_count"] = (df[car_cols] % 2).sum(axis=1)
    df["even_count"] = 3 - df["odd_count"]
    df["all_odd"] = (df["odd_count"] == 3).astype(int)
    df["all_even"] = (df["even_count"] == 3).astype(int)

    # 3. 人気の分散度（車番のばらつき）
    df["car_variance"] = df[car_cols].var(axis=1)

    # 4. 大穴指標（外枠が多いほど高い）
    df["outer_count"] = (df[car_cols] >= 7).sum(axis=1)
    df["inner_count"] = (df[car_cols] <= 3).sum(axis=1)

    # 5. 車番の積
    df["car_product"] = df["pos1_car_no"] * df["pos2_car_no"] * df["pos3_car_no"]

    # 6. 車番の差の絶対値
    df["diff_12"] = (df["pos1_car_no"] - df["pos2_car_no"]).abs()
    df["diff_23"] = (df["pos2_car_no"] - df["pos3_car_no"]).abs()
    df["diff_13"] = (df["pos1_car_no"] - df["pos3_car_no"]).abs()
    df["total_diff"] = df["diff_12"] + df["diff_23"] + df["diff_13"]

    # 7. 車番パターン（昇順・降順）
    df["is_ascending"] = (
        (df["pos1_car_no"] < df["pos2_car_no"]) &
        (df["pos2_car_no"] < df["pos3_car_no"])
    ).astype(int)
    df["is_descending"] = (
        (df["pos1_car_no"] > df["pos2_car_no"]) &
        (df["pos2_car_no"] > df["pos3_car_no"])
    ).astype(int)

    # 8. レース番号を数値化
    df["race_no_numeric"] = pd.to_numeric(
        df["race_no"].str.upper().str.replace("R", "", regex=False),
        errors="coerce"
    ).fillna(0)

    return df


def build_dataset(csv_path: Path) -> tuple:
    """データセットを構築"""
    df = pd.read_csv(csv_path)
    df["trifecta_payout_value"] = df["trifecta_payout"].apply(parse_payout)
    df = df.dropna(subset=["trifecta_payout_value"])
    df["high_payout"] = (df["trifecta_payout_value"] >= 10000).astype(int)

    # 高度な特徴量を生成
    df = create_advanced_features(df)

    # カテゴリカル列の処理（シンプルに）
    cat_cols = ["grade", "track", "category"]
    for col in cat_cols:
        df[col] = df[col].fillna("(欠損)").astype(str)

    # Label Encoding（LightGBMのため）
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 数値特徴量
    numeric_cols = [
        "car_sum", "car_std", "car_range", "car_median", "car_min", "car_max", "car_mean",
        "car_variance", "outer_count", "inner_count", "car_product",
        "diff_12", "diff_23", "diff_13", "total_diff",
        "is_consecutive", "odd_count", "even_count", "all_odd", "all_even",
        "is_ascending", "is_descending", "race_no_numeric",
        "pos1_car_no", "pos2_car_no", "pos3_car_no"
    ]

    # カテゴリカル特徴量（エンコード済み）
    cat_encoded_cols = [f"{col}_encoded" for col in cat_cols]

    # 統計情報を保存（予測時に使用）
    stats = {
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "cat_encoded_cols": cat_encoded_cols,
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()}
    }

    # 特徴量とターゲット
    feature_cols = numeric_cols + cat_encoded_cols
    X = df[feature_cols]
    y = df["high_payout"].values

    return X, y, df, stats


def train_model(X: pd.DataFrame, y: np.ndarray):
    """LightGBMモデルを学習"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LightGBMのパラメータ（調整済み）
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    }

    # データセット作成
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # モデル学習
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # 予測
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 評価指標
    metrics = {
        "auc": float(roc_auc_score(y_test, y_pred_proba)),
        "average_precision": float(average_precision_score(y_test, y_pred_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "positive_rate": float(y.mean()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    metrics["feature_importance"] = feature_importance.head(20).to_dict('records')

    return model, metrics


def main():
    # データの読み込み
    csv_path = Path("data/keirin_results_20240101_20251004.csv")
    if not csv_path.exists():
        raise SystemExit(f"データファイルが見つかりません: {csv_path}")

    print("=" * 60)
    print("高精度競輪予測モデルの学習")
    print("=" * 60)

    print("\n[1/4] データセットを構築中...")
    X, y, df, stats = build_dataset(csv_path)

    print(f"  総レース数: {len(df):,}")
    print(f"  高配当レース数: {df['high_payout'].sum():,} ({df['high_payout'].mean()*100:.1f}%)")
    print(f"  特徴量数: {X.shape[1]}")

    print("\n[2/4] LightGBMモデルを学習中...")
    model, metrics = train_model(X, y)

    print(f"\n[3/4] モデル評価:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  精度（全体）: {metrics['classification_report']['accuracy']:.4f}")
    print(f"  再現率（高配当）: {metrics['classification_report']['1']['recall']:.4f}")
    print(f"  適合率（高配当）: {metrics['classification_report']['1']['precision']:.4f}")

    print("\n  特徴量重要度（上位10）:")
    for item in metrics['feature_importance'][:10]:
        print(f"    {item['feature']}: {item['importance']:.0f}")

    # モデルとパラメータを保存
    model_dir = Path("backend/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n[4/4] モデルを保存中...")

    # モデルの保存（LightGBM専用形式）
    model.save_model(str(model_dir / "model_lgb.txt"))

    # Pickleでも保存（互換性のため）
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 統計情報の保存
    with open(model_dir / "model_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # メトリクスの保存
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # リファレンスデータ（UIで使用）
    reference_data = {
        "tracks": sorted(df["track"].unique().tolist()),
        "grades": sorted(df["grade"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()),
    }

    with open(model_dir / "reference_data.json", "w", encoding="utf-8") as f:
        json.dump(reference_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完了！モデルを保存しました: {model_dir}")
    print("  - model_lgb.txt (LightGBMモデル)")
    print("  - model.pkl (Pickleバックアップ)")
    print("  - model_stats.json (特徴量情報)")
    print("  - metrics.json (評価指標)")
    print("  - reference_data.json (リファレンスデータ)")

    print("\n" + "=" * 60)
    print("🎉 高精度モデルの学習が完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
