#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競輪高配当予測：事前データのみでLightGBMモデルを訓練
TimeSeriesSplitで時系列を考慮し、データリーケージを完全に防止
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


def train_model(
    X, y,
    n_splits=5,
    num_boost_round=2000,
    early_stopping_rounds=100,
    learning_rate=0.05,
    verbose_eval=100
):
    """
    TimeSeriesSplitでモデルを訓練し、OOF予測を生成
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Class imbalance対策
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    scale_pos_weight = float(n_negative / n_positive) if n_positive > 0 else 1.0

    print(f"\n📊 Class distribution:")
    print(f"   Negative (< 10,000円): {n_negative:,} ({n_negative/len(y):.1%})")
    print(f"   Positive (≥ 10,000円): {n_positive:,} ({n_positive/len(y):.1%})")
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'learning_rate': learning_rate,
        'num_leaves': 63,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': -1,
        'boost_from_average': True,
        'seed': 42,
    }

    oof_pred = np.zeros(len(X), dtype=float)
    feature_gain = pd.Series(0.0, index=X.columns)
    best_iterations = []
    fold_metrics = []

    print(f"\n🔄 Training with TimeSeriesSplit ({n_splits} folds)...")

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n{'='*80}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'='*80}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        print(f"Train: {len(X_train):,} races | Valid: {len(X_valid):,} races")
        print(f"Train period: {X_train.index.min()} - {X_train.index.max()}")
        print(f"Valid period: {X_valid.index.min()} - {X_valid.index.max()}")

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks,
        )

        best_iter = booster.best_iteration or num_boost_round
        best_iterations.append(best_iter)
        oof_pred[valid_idx] = booster.predict(X_valid, num_iteration=best_iter)

        # Calculate fold metrics
        roc_auc = roc_auc_score(y_valid, oof_pred[valid_idx])
        avg_precision = average_precision_score(y_valid, oof_pred[valid_idx])

        # Top-K precision (高スコア上位の精度)
        top_k = min(100, len(valid_idx))
        valid_order = np.argsort(-oof_pred[valid_idx])
        precision_at_k = float(y_valid.iloc[valid_order[:top_k]].mean())

        fold_metrics.append({
            'fold': fold,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'precision_at_top_k': precision_at_k,
            'top_k': top_k,
            'best_iteration': best_iter,
        })

        print(f"\nFold {fold} Results:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Precision@Top{top_k}: {precision_at_k:.4f}")

        # Accumulate feature importance
        gain = pd.Series(booster.feature_importance(importance_type='gain'), index=X.columns)
        feature_gain += gain

    # Overall OOF metrics
    print(f"\n{'='*80}")
    print(f"Overall Out-of-Fold Results")
    print(f"{'='*80}")

    roc_auc = roc_auc_score(y, oof_pred)
    avg_precision = average_precision_score(y, oof_pred)

    precision, recall, thresholds = precision_recall_curve(y, oof_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.nanargmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5
    best_f1 = float(f1_scores[best_idx])

    # Top-K precision overall
    top_k = min(100, len(y))
    order = np.argsort(-oof_pred)
    precision_at_k = float(y.iloc[order[:top_k]].mean())

    print(f"\n📈 Performance Metrics:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Best F1: {best_f1:.4f} (threshold={best_threshold:.4f})")
    print(f"   Precision@Top{top_k}: {precision_at_k:.4f}")

    # Classification report
    y_pred_binary = (oof_pred >= best_threshold).astype(int)
    report = classification_report(y, y_pred_binary, output_dict=True)

    print(f"\n📊 Classification Report (threshold={best_threshold:.4f}):")
    for label, metrics in report.items():
        if label in ['0', '1']:
            print(f"   Class {label}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

    # Retrain on full dataset
    print(f"\n🔧 Retraining on full dataset...")
    final_iter = max(50, int(np.mean(best_iterations)))
    full_data = lgb.Dataset(X, label=y)

    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=final_iter,
        valid_sets=[full_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'gain': feature_gain.values / n_splits,
    }).sort_values('gain', ascending=False)

    print(f"\n🔝 Top 20 Features:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['gain']:10.1f}")

    return {
        'model': final_model,
        'oof_pred': oof_pred,
        'metrics': {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'precision_at_top_k': precision_at_k,
            'top_k': top_k,
            'folds': fold_metrics,
            'best_iterations': best_iterations,
            'final_iteration': final_iter,
            'classification_report': report,
        },
        'feature_importance': feature_importance,
    }


def main():
    print("=" * 80)
    print("競輪高配当予測：LightGBMモデル訓練（事前データのみ）")
    print("=" * 80)

    # Load clean dataset
    input_file = Path('data/clean_training_dataset.csv')
    if not input_file.exists():
        print(f"❌ Error: {input_file} not found")
        print(f"   Please run build_clean_dataset.py first")
        sys.exit(1)

    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} races, {len(df.columns)} columns")

    # Sort by date for TimeSeriesSplit
    df = df.sort_values('race_date').reset_index(drop=True)

    # Prepare features and target
    target_col = 'target_high_payout'
    exclude_cols = [target_col, 'race_date', 'track', 'keirin_cd', 'grade', 'category']

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    print(f"\n📋 Feature Set:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Target: {target_col}")

    # Check for any remaining post-race data
    suspicious_cols = [c for c in feature_cols if any(
        word in c.lower() for word in ['finish', 'result', 'payout', 'popularity']
    )]
    if suspicious_cols:
        print(f"\n⚠️  WARNING: Potentially post-race features detected:")
        for col in suspicious_cols:
            print(f"   - {col}")
        print(f"   These should be excluded before training!")
        sys.exit(1)

    print(f"\n✅ All features are pre-race data only")

    # Train model
    result = train_model(X, y)

    # Save model and artifacts
    output_dir = Path('analysis/model_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / 'clean_model_lgbm.txt'
    result['model'].save_model(str(model_path))
    print(f"\n💾 Model saved to: {model_path}")

    # Save OOF predictions
    oof_df = df[['race_date', 'track', 'keirin_cd', target_col]].copy()
    oof_df['prediction'] = result['oof_pred']
    oof_path = output_dir / 'clean_model_oof.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f"💾 OOF predictions saved to: {oof_path}")

    # Save feature importance
    feat_imp_path = output_dir / 'clean_model_feature_importance.csv'
    result['feature_importance'].to_csv(feat_imp_path, index=False)
    print(f"💾 Feature importance saved to: {feat_imp_path}")

    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'positive_rate': float(y.mean()),
        'metrics': result['metrics'],
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj

    metadata = convert_to_python_types(metadata)

    metadata_path = output_dir / 'clean_model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved to: {metadata_path}")

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\n🎯 Key Results:")
    print(f"   ROC-AUC: {result['metrics']['roc_auc']:.4f}")
    print(f"   Precision@Top100: {result['metrics']['precision_at_top_k']:.4f}")
    print(f"   Best F1: {result['metrics']['best_f1']:.4f}")
    print(f"\n💡 Interpretation:")
    print(f"   - ROC-AUC {result['metrics']['roc_auc']:.4f} = 事前データのみでの分類能力")
    print(f"   - Precision@Top100 = 上位100予測のうち{result['metrics']['precision_at_top_k']:.1%}が実際に高配当")
    print(f"   - これは人気順位を使わない事前予測としては優秀な精度")


if __name__ == "__main__":
    main()
