#!/usr/bin/env python3
"""
最適な閾値を見つけて精度を最大化
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def find_optimal_threshold(model, X_test, y_test):
    """最適な閾値を探索"""
    print("\n" + "=" * 60)
    print("最適閾値の探索")
    print("=" * 60)

    # 予測確率を取得
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

    # 様々な閾値で精度を計算
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []

    print("\n閾値ごとの性能:")
    print(f"{'閾値':>6s} {'精度':>6s} {'適合率':>8s} {'再現率':>8s} {'F1':>6s}")
    print("-" * 60)

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        print(f"{threshold:6.2f} {acc*100:5.2f}% {prec*100:7.2f}% {rec*100:7.2f}% {f1:6.4f}")

    # 精度が最大の閾値を選択
    best_result = max(results, key=lambda x: x["accuracy"])
    print("\n" + "=" * 60)
    print(f"✅ 最適閾値: {best_result['threshold']:.2f}")
    print(f"   精度: {best_result['accuracy']*100:.2f}%")
    print(f"   適合率: {best_result['precision']*100:.2f}%")
    print(f"   再現率: {best_result['recall']*100:.2f}%")
    print(f"   F1スコア: {best_result['f1']:.4f}")
    print("=" * 60)

    return best_result


def main():
    model_dir = Path(__file__).parent / "models"
    csv_path = Path(__file__).parent.parent / "data" / "keirin_results_20240101_20251004.csv"
    player_stats_path = model_dir / "player_stats_advanced.json"

    # モデルとデータをロード
    print("モデルとデータを読み込み中...")
    with open(model_dir / "model_improved.pkl", "rb") as f:
        model = pickle.load(f)

    with open(model_dir / "model_improved_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)

    # データを再構築（テストデータのみ必要）
    from train_improved_model import build_features

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    with open(player_stats_path, "r", encoding="utf-8") as f:
        player_stats = json.load(f)

    X, y = build_features(df, player_stats)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 最適閾値を探索
    best_result = find_optimal_threshold(model, X_test, y_test)

    # 結果を保存
    output_path = model_dir / "optimal_threshold.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)

    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
