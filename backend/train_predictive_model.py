#!/usr/bin/env python3
"""
レース前予測可能なモデルを学習
出走選手の過去成績を使用
"""
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


def parse_payout(value: str) -> float:
    """配当金文字列を数値に変換"""
    if pd.isna(value) or value == "":
        return np.nan
    digits = re.sub(r"[^0-9]", "", str(value))
    return float(digits) if digits else np.nan


def get_player_features(player_name: str, player_stats: dict) -> dict:
    """選手の過去成績から特徴量を取得"""
    if player_name in player_stats:
        stats = player_stats[player_name]
        return {
            "win_rate": stats["win_rate"],
            "place_2_rate": stats["place_2_rate"],
            "place_3_rate": stats["place_3_rate"],
            "top3_rate": stats["top3_rate"],
            "avg_payout": stats["avg_payout"],
            "high_payout_rate": stats["high_payout_rate"],
            "races": min(stats["races"], 500) / 500,  # 正規化
        }
    else:
        # 未知の選手はデフォルト値
        return {
            "win_rate": 0.1,
            "place_2_rate": 0.1,
            "place_3_rate": 0.1,
            "top3_rate": 0.3,
            "avg_payout": 5000,
            "high_payout_rate": 0.2,
            "races": 0.0,
        }


def build_features(df: pd.DataFrame, player_stats: dict) -> tuple:
    """レースごとに特徴量を構築"""

    print("\n特徴量を構築中...")

    features_list = []
    targets = []

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  処理中: {idx}/{len(df)} レース")

        # ターゲット
        trifecta_payout = row.get("trifecta_payout_value", 0)
        if pd.isna(trifecta_payout):
            continue

        target = 1 if trifecta_payout >= 10000 else 0

        # 基本情報
        track = row.get("track", "")
        grade = row.get("grade", "")
        category = row.get("category", "")

        # 1-2-3着選手の統計
        pos1_name = row.get("pos1_name", "")
        pos2_name = row.get("pos2_name", "")
        pos3_name = row.get("pos3_name", "")

        pos1_stats = get_player_features(pos1_name, player_stats)
        pos2_stats = get_player_features(pos2_name, player_stats)
        pos3_stats = get_player_features(pos3_name, player_stats)

        # 車番（NaN対策）
        pos1_car = row.get("pos1_car_no", 5)
        pos2_car = row.get("pos2_car_no", 5)
        pos3_car = row.get("pos3_car_no", 5)

        if pd.isna(pos1_car):
            pos1_car = 5
        if pd.isna(pos2_car):
            pos2_car = 5
        if pd.isna(pos3_car):
            pos3_car = 5

        pos1_car = int(pos1_car)
        pos2_car = int(pos2_car)
        pos3_car = int(pos3_car)

        # 特徴量を構築
        features = {
            # 選手統計（1着）
            "pos1_win_rate": pos1_stats["win_rate"],
            "pos1_top3_rate": pos1_stats["top3_rate"],
            "pos1_avg_payout": pos1_stats["avg_payout"],
            "pos1_high_payout_rate": pos1_stats["high_payout_rate"],

            # 選手統計（2着）
            "pos2_win_rate": pos2_stats["win_rate"],
            "pos2_top3_rate": pos2_stats["top3_rate"],
            "pos2_avg_payout": pos2_stats["avg_payout"],
            "pos2_high_payout_rate": pos2_stats["high_payout_rate"],

            # 選手統計（3着）
            "pos3_win_rate": pos3_stats["win_rate"],
            "pos3_top3_rate": pos3_stats["top3_rate"],
            "pos3_avg_payout": pos3_stats["avg_payout"],
            "pos3_high_payout_rate": pos3_stats["high_payout_rate"],

            # 3選手の統計的特徴
            "avg_win_rate": np.mean([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "std_win_rate": np.std([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "min_win_rate": np.min([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "max_win_rate": np.max([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),

            # 車番特徴
            "pos1_car_no": pos1_car,
            "pos2_car_no": pos2_car,
            "pos3_car_no": pos3_car,
            "car_sum": pos1_car + pos2_car + pos3_car,
            "car_std": np.std([pos1_car, pos2_car, pos3_car]),
            "car_range": max(pos1_car, pos2_car, pos3_car) - min(pos1_car, pos2_car, pos3_car),

            # 外枠・内枠
            "outer_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c >= 7),
            "inner_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c <= 3),

            # カテゴリカル（簡易エンコード）
            "is_F1": 1 if grade == "F1" else 0,
            "is_F2": 1 if grade == "F2" else 0,
            "is_G1": 1 if grade == "G1" else 0,
            "is_G2": 1 if grade == "G2" else 0,
            "is_G3": 1 if grade == "G3" else 0,
        }

        features_list.append(features)
        targets.append(target)

    print(f"  完了: {len(features_list)} レース")

    return pd.DataFrame(features_list), np.array(targets)


def train_model(X: pd.DataFrame, y: np.array):
    """LightGBMモデルを学習"""

    print("\nモデルを学習中...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

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

    # 評価
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_pred_proba)),
        "average_precision": float(average_precision_score(y_test, y_pred_proba)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    return model, metrics, feature_importance


def main():
    print("=" * 60)
    print("レース前予測可能なモデルの学習")
    print("=" * 60)

    # データ読み込み
    csv_path = Path("data/keirin_results_20240101_20251004.csv")
    player_stats_path = Path("backend/models/player_stats.json")

    if not csv_path.exists():
        raise SystemExit(f"データが見つかりません: {csv_path}")
    if not player_stats_path.exists():
        raise SystemExit(f"選手統計が見つかりません: {player_stats_path}\n"
                         f"先に build_player_stats.py を実行してください")

    print("\n[1/4] データを読み込み中...")
    df = pd.read_csv(csv_path)
    df["trifecta_payout_value"] = df["trifecta_payout"].apply(parse_payout)
    df = df.dropna(subset=["trifecta_payout_value"])

    with open(player_stats_path, "r", encoding="utf-8") as f:
        player_stats = json.load(f)

    print(f"  レース数: {len(df):,}")
    print(f"  選手数: {len(player_stats):,}")

    # 特徴量構築
    print("\n[2/4] 特徴量を構築中...")
    X, y = build_features(df, player_stats)

    print(f"\n  特徴量数: {X.shape[1]}")
    print(f"  高配当レース: {y.sum():,} / {len(y):,} ({y.mean()*100:.1f}%)")

    # モデル学習
    print("\n[3/4] LightGBMモデルを学習中...")
    model, metrics, feature_importance = train_model(X, y)

    print(f"\n  AUC: {metrics['auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  精度: {metrics['classification_report']['accuracy']:.4f}")

    print("\n  特徴量重要度Top 10:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:25s} {row['importance']:8.0f}")

    # 保存
    print("\n[4/4] モデルを保存中...")
    model_dir = Path("backend/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / "model_predictive.txt"))

    with open(model_dir / "model_predictive.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / "metrics_predictive.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model_info = {
        "feature_names": list(X.columns),
        "feature_count": X.shape[1],
    }

    with open(model_dir / "model_predictive_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完了！")
    print(f"  - model_predictive.txt")
    print(f"  - model_predictive.pkl")
    print(f"  - metrics_predictive.json")
    print(f"  - model_predictive_info.json")

    print("\n" + "=" * 60)
    print("🎉 レース前予測可能なモデルが完成しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
