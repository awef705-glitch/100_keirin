#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
事前予測モデルの有益性評価

人気順位を使わず、事前情報のみでどれくらい的中するかを検証
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve

import train_complete_prerace_model as complete_model

MODEL_DIR = Path("analysis") / "model_outputs"


def evaluate_simple_baseline(dataset: pd.DataFrame, y: pd.Series) -> dict:
    """
    単純なベースライン評価

    1. ランダム予測
    2. 多数派予測（常に「高配当でない」と予測）
    3. 実力格差のみ（tokuten_range > threshold）
    """
    results = {}

    # ランダム予測
    random_pred = np.random.rand(len(y))
    results["random_auc"] = roc_auc_score(y, random_pred)
    results["random_accuracy"] = 0.5

    # 多数派予測
    majority_pred = np.zeros(len(y))
    results["majority_accuracy"] = (y == 0).mean()

    # 実力格差のみ（簡易ルールベース）
    if "tokuten_range" in dataset.columns:
        tokuten_range = dataset["tokuten_range"].fillna(0)
        # 実力差が大きいレースを「高配当」と予測
        threshold = tokuten_range.quantile(0.75)
        rule_based_pred = (tokuten_range > threshold).astype(int)

        if rule_based_pred.sum() > 0:
            precision = (rule_based_pred & y).sum() / rule_based_pred.sum()
            recall = (rule_based_pred & y).sum() / y.sum()
            results["rule_based_precision"] = precision
            results["rule_based_recall"] = recall
            results["rule_based_f1"] = 2 * precision * recall / (precision + recall + 1e-9)

    return results


def train_and_evaluate_prerace_model(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list,
    n_splits: int = 5,
) -> dict:
    """
    事前予測モデルを訓練・評価

    TimeSeriesSplitで交差検証し、以下を計算:
    - 的中率（Accuracy）
    - 精度（Precision）
    - 再現率（Recall）
    - ROC-AUC
    - Top-K精度
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_predictions = np.zeros(len(X))
    fold_metrics = []

    print(f"\n交差検証開始 ({n_splits}-fold)...")

    for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # カテゴリ変数を数値に変換（HistGradientBoostingClassifierはカテゴリ未対応）
        for col in categorical_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(str).astype('category').cat.codes
                X_valid[col] = X_valid[col].astype(str).astype('category').cat.codes

        # モデル訓練
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # 予測
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        oof_predictions[valid_idx] = y_pred_proba

        # メトリクス計算
        auc = roc_auc_score(y_valid, y_pred_proba)
        report = classification_report(y_valid, y_pred, output_dict=True)

        fold_metrics.append({
            "fold": fold_idx,
            "auc": auc,
            "accuracy": report["accuracy"],
            "precision": report.get("1", {}).get("precision", 0),
            "recall": report.get("1", {}).get("recall", 0),
            "f1": report.get("1", {}).get("f1-score", 0),
        })

        print(f"  Fold {fold_idx}: AUC={auc:.4f}, Accuracy={report['accuracy']:.4f}, "
              f"Precision={report.get('1', {}).get('precision', 0):.4f}")

    # 全体のメトリクス
    overall_auc = roc_auc_score(y, oof_predictions)
    overall_pred = (oof_predictions >= 0.5).astype(int)
    overall_report = classification_report(y, overall_pred, output_dict=True)

    # Top-K精度
    top_k_metrics = {}
    for k in [10, 50, 100, 200, 500]:
        if k <= len(oof_predictions):
            top_k_idx = np.argsort(-oof_predictions)[:k]
            precision_at_k = y.iloc[top_k_idx].mean()
            top_k_metrics[f"precision@{k}"] = precision_at_k

    # 閾値別の精度
    precisions, recalls, thresholds = precision_recall_curve(y, oof_predictions)

    # 最適閾値（F1最大）
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]

    return {
        "fold_metrics": fold_metrics,
        "overall_auc": overall_auc,
        "overall_accuracy": overall_report["accuracy"],
        "overall_precision": overall_report.get("1", {}).get("precision", 0),
        "overall_recall": overall_report.get("1", {}).get("recall", 0),
        "overall_f1": overall_report.get("1", {}).get("f1-score", 0),
        "top_k_metrics": top_k_metrics,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "oof_predictions": oof_predictions,
    }


def generate_performance_report(
    baseline_results: dict,
    model_results: dict,
    y: pd.Series,
) -> str:
    """パフォーマンスレポートを生成"""

    positive_rate = y.mean()

    report = []
    report.append("=" * 80)
    report.append("事前予測モデルの有益性評価")
    report.append("=" * 80)

    report.append(f"\nデータ概要:")
    report.append(f"  総レース数:       {len(y):,}")
    report.append(f"  高配当レース数:   {y.sum():,} ({positive_rate*100:.2f}%)")
    report.append(f"  非高配当レース数: {(~y.astype(bool)).sum():,} ({(1-positive_rate)*100:.2f}%)")

    report.append(f"\n" + "-" * 80)
    report.append("ベースライン比較")
    report.append("-" * 80)

    report.append(f"\n1. ランダム予測:")
    report.append(f"   ROC-AUC:  {baseline_results.get('random_auc', 0):.4f}")
    report.append(f"   精度:     50%（期待値）")

    report.append(f"\n2. 多数派予測（常に「高配当でない」）:")
    report.append(f"   精度:     {baseline_results.get('majority_accuracy', 0)*100:.2f}%")
    report.append(f"   ※ 高配当を一切当てられない")

    if "rule_based_precision" in baseline_results:
        report.append(f"\n3. ルールベース（実力格差のみ）:")
        report.append(f"   精度:     {baseline_results['rule_based_precision']*100:.2f}%")
        report.append(f"   再現率:   {baseline_results['rule_based_recall']*100:.2f}%")
        report.append(f"   F1スコア: {baseline_results['rule_based_f1']:.4f}")

    report.append(f"\n" + "-" * 80)
    report.append("事前予測モデルの性能")
    report.append("-" * 80)

    report.append(f"\nROC-AUC:      {model_results['overall_auc']:.4f}")
    report.append(f"精度:         {model_results['overall_accuracy']*100:.2f}%")
    report.append(f"適合率:       {model_results['overall_precision']*100:.2f}%")
    report.append(f"再現率:       {model_results['overall_recall']*100:.2f}%")
    report.append(f"F1スコア:     {model_results['overall_f1']:.4f}")
    report.append(f"最適閾値:     {model_results['best_threshold']:.4f}")
    report.append(f"最大F1:       {model_results['best_f1']:.4f}")

    report.append(f"\n" + "-" * 80)
    report.append("Top-K精度（高スコアのレースに絞った場合の的中率）")
    report.append("-" * 80)

    for k, precision in sorted(model_results["top_k_metrics"].items()):
        k_num = int(k.split("@")[1])
        report.append(f"\nTop {k_num:3d} レース: {precision*100:.2f}% 的中")
        expected_hits = int(k_num * precision)
        report.append(f"  → {k_num}レース中 約{expected_hits}レースが高配当")

    report.append(f"\n" + "=" * 80)
    report.append("実用的な解釈")
    report.append("=" * 80)

    # Top 100の精度
    top100_precision = model_results["top_k_metrics"].get("precision@100", 0)

    report.append(f"\n【投資戦略への応用】")
    report.append(f"\n1. モデルが最も自信を持つTop 100レースに絞った場合:")
    report.append(f"   的中率: {top100_precision*100:.2f}%")
    report.append(f"   期待的中数: 100レース中 約{int(100*top100_precision)}レース")

    if top100_precision > positive_rate:
        improvement = (top100_precision / positive_rate - 1) * 100
        report.append(f"   ランダムより: {improvement:.1f}% 向上")

    report.append(f"\n2. 実用例:")
    report.append(f"   - 毎日のレースから高スコアのものだけに投資")
    report.append(f"   - 無駄な投資を削減")
    report.append(f"   - 長期的に優位性を得る")

    report.append(f"\n3. 限界:")
    report.append(f"   - 完璧な予測は不可能（市場の集合知を超えるのは困難）")
    report.append(f"   - 的中率{top100_precision*100:.1f}%でも、配当次第で収支はプラス・マイナス")
    report.append(f"   - リスク管理が重要")

    report.append(f"\n" + "=" * 80)

    return "\n".join(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="事前予測モデルの有益性評価")

    parser.add_argument("--results", default=complete_model.DATA_DIR / "keirin_results_20240101_20251004.csv", type=Path)
    parser.add_argument("--prerace", default=complete_model.DATA_DIR / "keirin_prerace_20240101_20251004.csv", type=Path)
    parser.add_argument("--entries", default=complete_model.DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv", type=Path)
    parser.add_argument("--threshold", default=10000, type=int)
    parser.add_argument("--folds", default=5, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("事前予測モデルの有益性評価")
    print("=" * 80)

    # データセット構築
    print("\nデータセット構築中...")
    dataset, X, y, numeric_features, categorical_features = complete_model.build_complete_dataset(
        args.results,
        args.prerace,
        args.entries,
        args.threshold,
    )

    # ベースライン評価
    print("\nベースライン評価中...")
    baseline_results = evaluate_simple_baseline(dataset, y)

    # モデル評価
    print("\nモデル訓練・評価中...")
    model_results = train_and_evaluate_prerace_model(
        X, y, categorical_features, n_splits=args.folds
    )

    # レポート生成
    report = generate_performance_report(baseline_results, model_results, y)
    print("\n" + report)

    # 結果保存
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    report_path = MODEL_DIR / "prerace_model_evaluation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✓ レポート保存: {report_path}")

    # メトリクス保存
    metrics = {
        "baseline": baseline_results,
        "model": {k: v for k, v in model_results.items() if k != "oof_predictions"},
        "data_summary": {
            "total_races": len(y),
            "high_payout_races": int(y.sum()),
            "high_payout_rate": float(y.mean()),
        },
    }

    metrics_path = MODEL_DIR / "prerace_model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✓ メトリクス保存: {metrics_path}")


if __name__ == "__main__":
    main()
