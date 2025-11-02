#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Advanced high-payout prediction model with enhanced features and ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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

import train_high_payout_model as base

MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "advanced_model_lgbm.txt"
METRICS_PATH = MODEL_DIR / "advanced_model_metrics.json"


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced feature engineering."""
    result = df.copy()

    # 日付関連の特徴量
    result['race_date_str'] = result['race_date'].astype(str)
    result['year'] = result['race_date_str'].str[:4].astype(int)
    result['month'] = result['race_date_str'].str[4:6].astype(int)
    result['day'] = result['race_date_str'].str[6:8].astype(int)

    # 曜日を計算（簡易版：ツェラーの公式の代わりにpandasを使用）
    result['date_obj'] = pd.to_datetime(result['race_date_str'], format='%Y%m%d', errors='coerce')
    result['day_of_week'] = result['date_obj'].dt.dayofweek
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    result['is_holiday_season'] = result['month'].isin([1, 5, 8, 12]).astype(int)

    # 季節
    result['season'] = ((result['month'] % 12) // 3 + 1)  # 1:春 2:夏 3:秋 4:冬

    # レース番号関連
    result['is_final_race'] = (result['race_no_int'] >= 11).astype(int)
    result['is_early_race'] = (result['race_no_int'] <= 3).astype(int)

    # 人気度関連（最重要特徴量）
    if 'trifecta_popularity' in result.columns:
        # 対数変換で外れ値の影響を軽減
        result['trifecta_popularity_log'] = np.log1p(result['trifecta_popularity'].fillna(0))
        # 人気度の二乗（非線形関係）
        result['trifecta_popularity_squared'] = result['trifecta_popularity'].fillna(0) ** 2
        # 人気度カテゴリ
        result['popularity_category'] = pd.cut(
            result['trifecta_popularity'].fillna(999),
            bins=[0, 3, 10, 30, 100, 999],
            labels=['very_popular', 'popular', 'medium', 'unpopular', 'very_unpopular']
        ).astype(str)

    # 得点のばらつきと集中度
    if 'heikinTokuten_mean' in result.columns and 'heikinTokuten_std' in result.columns:
        # 変動係数が既にある場合も、追加の統計量を計算
        result['heikinTokuten_skew_proxy'] = (
            result['heikinTokuten_mean'] - result['heikinTokuten_min']
        ) / (result['heikinTokuten_max'] - result['heikinTokuten_min'] + 1e-6)

        # 得点の集中度（逆CV）
        result['heikinTokuten_concentration'] = 1.0 / (result['heikinTokuten_cv'].fillna(1) + 0.01)

    # 脚質の多様性
    style_ratio_cols = [col for col in result.columns if col.endswith('_ratio')]
    if len(style_ratio_cols) >= 2:
        # ジニ係数風の不均等度指標
        style_ratios = result[style_ratio_cols].fillna(0)
        result['style_diversity'] = 1.0 - (style_ratios ** 2).sum(axis=1)

        # 最大脚質の割合
        result['style_max_ratio'] = style_ratios.max(axis=1)
        result['style_min_ratio'] = style_ratios.min(axis=1)

    # インタラクション特徴量
    if 'entry_count' in result.columns and 'heikinTokuten_std' in result.columns:
        result['entry_x_tokuten_std'] = result['entry_count'] * result['heikinTokuten_std']

    if 'trifecta_popularity' in result.columns and 'heikinTokuten_cv' in result.columns:
        result['popularity_x_cv'] = result['trifecta_popularity'].fillna(0) * result['heikinTokuten_cv'].fillna(0)

    # 出走数の統計
    count_cols = ['nigeCnt_mean', 'makuriCnt_mean', 'sasiCnt_mean', 'markCnt_mean', 'backCnt_mean']
    existing_count_cols = [col for col in count_cols if col in result.columns]
    if len(existing_count_cols) >= 2:
        result['total_tactics_count'] = result[existing_count_cols].sum(axis=1)
        result['tactics_entropy'] = 0
        for col in existing_count_cols:
            p = result[col] / (result['total_tactics_count'] + 1e-6)
            result['tactics_entropy'] -= p * np.log(p + 1e-6)

    # レース会場の難易度代理指標（会場別の高配当率は後で計算）
    # ここでは簡易的に、会場コードとカテゴリの組み合わせを特徴量化
    if 'keirin_cd' in result.columns and 'category' in result.columns:
        result['venue_category'] = result['keirin_cd'].astype(str) + '_' + result['category'].astype(str)

    # 欠損値の数（データ品質の指標）
    result['missing_count'] = result.isnull().sum(axis=1)

    return result


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str = 'target_high_payout',
    smoothing: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add target encoding for categorical variables."""
    train_result = train_df.copy()
    test_result = test_df.copy()

    global_mean = train_df[target_col].mean()

    for col in categorical_cols:
        if col not in train_df.columns or col not in test_df.columns:
            continue

        # 訓練データでの集計
        agg = train_df.groupby(col)[target_col].agg(['sum', 'count'])

        # スムージング付きターゲットエンコーディング
        smoothed_mean = (agg['sum'] + smoothing * global_mean) / (agg['count'] + smoothing)

        # 新しい列名
        encoding_col = f'{col}_target_enc'

        # マッピング
        train_result[encoding_col] = train_df[col].map(smoothed_mean).fillna(global_mean)
        test_result[encoding_col] = test_df[col].map(smoothed_mean).fillna(global_mean)

    return train_result, test_result


def build_advanced_dataset(
    results_path: Path,
    prerace_path: Path,
    entries_path: Path,
    payout_threshold: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build dataset with advanced features."""
    # 基本データセット構築
    results = base.load_results(results_path, payout_threshold)
    prerace = base.load_prerace(prerace_path)
    entries = base.aggregate_entries(entries_path)
    dataset = base.merge_datasets(results, prerace, entries)

    # 既存の派生特徴量
    dataset = base.add_derived_features(dataset)

    # 高度な特徴量追加
    dataset = add_advanced_features(dataset)

    # 時系列でソート
    dataset = dataset.sort_values(['race_date', 'keirin_cd', 'race_no_int']).reset_index(drop=True)

    # 特徴量選択
    numeric_features, categorical_features = base.select_feature_columns(dataset)

    # 追加の数値特徴量
    additional_numeric = [
        'year', 'month', 'day', 'day_of_week', 'is_weekend', 'is_holiday_season',
        'season', 'is_final_race', 'is_early_race',
        'trifecta_popularity_log', 'trifecta_popularity_squared',
        'heikinTokuten_skew_proxy', 'heikinTokuten_concentration',
        'style_diversity', 'style_max_ratio', 'style_min_ratio',
        'entry_x_tokuten_std', 'popularity_x_cv',
        'total_tactics_count', 'tactics_entropy', 'missing_count'
    ]

    # 追加のカテゴリ特徴量
    additional_categorical = ['popularity_category', 'venue_category']

    # 存在する特徴量のみ追加
    for feat in additional_numeric:
        if feat in dataset.columns and feat not in numeric_features:
            numeric_features.append(feat)

    for feat in additional_categorical:
        if feat in dataset.columns and feat not in categorical_features:
            categorical_features.append(feat)

    return dataset, numeric_features, categorical_features


def train_ensemble_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    n_splits: int = 5,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 100,
) -> Tuple[Dict, np.ndarray, List[lgb.Booster]]:
    """Train ensemble of LightGBM models with different parameters."""

    # 3つの異なるパラメータセット
    param_sets = [
        {  # バランス型
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 95,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "class_weight": "balanced",
        },
        {  # 深い木
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.02,
            "num_leaves": 127,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_data_in_leaf": 30,
            "lambda_l1": 0.05,
            "lambda_l2": 0.05,
            "verbose": -1,
            "class_weight": "balanced",
        },
        {  # 正則化強
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.04,
            "num_leaves": 63,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 80,
            "lambda_l1": 0.3,
            "lambda_l2": 0.3,
            "verbose": -1,
            "class_weight": "balanced",
        },
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 各パラメータセットでの予測を保存
    oof_preds = []
    all_models = []

    for param_idx, params in enumerate(param_sets):
        print(f"\nTraining model {param_idx + 1}/3...")
        oof_pred = np.zeros(len(X))
        fold_models = []

        for fold_idx, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)
            valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features, reference=train_set)

            callbacks = [
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100)
            ]

            booster = lgb.train(
                params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[train_set, valid_set],
                valid_names=['train', 'valid'],
                callbacks=callbacks,
            )

            best_iter = booster.best_iteration or num_boost_round
            y_valid_pred = booster.predict(X_valid, num_iteration=best_iter)
            oof_pred[valid_idx] = y_valid_pred
            fold_models.append(booster)

            print(f"  Fold {fold_idx}: Best iteration = {best_iter}, "
                  f"Valid AUC = {roc_auc_score(y_valid, y_valid_pred):.4f}")

        oof_preds.append(oof_pred)
        all_models.append(fold_models)

        # 個別モデルのスコア
        roc_auc = roc_auc_score(y, oof_pred)
        ap = average_precision_score(y, oof_pred)
        print(f"Model {param_idx + 1} OOF - ROC-AUC: {roc_auc:.4f}, Avg Precision: {ap:.4f}")

    # アンサンブル予測（平均）
    oof_ensemble = np.mean(oof_preds, axis=0)

    # 重み付き平均も試す（各モデルのAUCで重み付け）
    weights = [roc_auc_score(y, pred) for pred in oof_preds]
    weights = np.array(weights) / sum(weights)
    oof_weighted = np.average(oof_preds, axis=0, weights=weights)

    # どちらが良いか選択
    auc_ensemble = roc_auc_score(y, oof_ensemble)
    auc_weighted = roc_auc_score(y, oof_weighted)

    if auc_weighted > auc_ensemble:
        final_oof = oof_weighted
        ensemble_method = 'weighted_average'
        print(f"\nUsing weighted average (AUC: {auc_weighted:.4f})")
    else:
        final_oof = oof_ensemble
        ensemble_method = 'simple_average'
        print(f"\nUsing simple average (AUC: {auc_ensemble:.4f})")

    # 最終メトリクス計算
    overall_roc_auc = roc_auc_score(y, final_oof)
    overall_ap = average_precision_score(y, final_oof)
    precision, recall, thresholds = precision_recall_curve(y, final_oof)
    overall_report = classification_report(y, (final_oof >= 0.5).astype(int), output_dict=True)

    metrics = {
        'oof_roc_auc': overall_roc_auc,
        'oof_average_precision': overall_ap,
        'oof_classification_report': overall_report,
        'precision_curve': precision.tolist(),
        'recall_curve': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'ensemble_method': ensemble_method,
        'model_weights': weights.tolist() if ensemble_method == 'weighted_average' else [1/3, 1/3, 1/3],
        'individual_model_aucs': [roc_auc_score(y, pred) for pred in oof_preds],
    }

    return metrics, final_oof, all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced high-payout model with ensemble.")
    parser.add_argument(
        "--results",
        default=base.DATA_DIR / "keirin_results_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument(
        "--prerace",
        default=base.DATA_DIR / "keirin_prerace_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument(
        "--entries",
        default=base.DATA_DIR / "keirin_race_detail_entries_20240101_20251004.csv",
        type=Path,
    )
    parser.add_argument("--threshold", default=10000, type=int)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--num-boost-round", default=1000, type=int)
    parser.add_argument("--early-stopping-rounds", default=100, type=int)
    parser.add_argument("--top-k", default=100, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building advanced dataset...")
    dataset, numeric_features, categorical_features = build_advanced_dataset(
        args.results,
        args.prerace,
        args.entries,
        args.threshold,
    )

    print(f"Dataset shape: {dataset.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    feature_columns = numeric_features + categorical_features
    X = dataset[feature_columns].copy()

    for col in categorical_features:
        X[col] = X[col].astype('category')

    y = dataset['target_high_payout'].astype(int)

    print(f"\nPositive class ratio: {y.mean():.4f}")
    print(f"Total samples: {len(y)}")

    print("\nTraining ensemble models...")
    metrics, oof_pred, models = train_ensemble_cv(
        X, y, categorical_features,
        n_splits=args.folds,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    # Top-K precision
    order = np.argsort(-oof_pred)
    top_k = min(args.top_k, len(order))
    top_indices = order[:top_k]
    precision_at_k = float(y.iloc[top_indices].mean())

    # Best F1 threshold
    thresholds_arr = np.array(metrics['thresholds'])
    precision_arr = np.array(metrics['precision_curve'])
    recall_arr = np.array(metrics['recall_curve'])

    if len(thresholds_arr):
        f1_scores = 2 * precision_arr[:-1] * recall_arr[:-1] / (
            precision_arr[:-1] + recall_arr[:-1] + 1e-9
        )
        best_idx = int(np.nanargmax(f1_scores))
        best_threshold = thresholds_arr[best_idx]
        best_f1 = f1_scores[best_idx]
    else:
        best_threshold = 0.5
        best_f1 = float('nan')

    # Save OOF predictions
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = dataset[['race_date', 'keirin_cd', 'race_no_int', 'target_high_payout']].copy()
    oof_df['prediction'] = oof_pred
    oof_df.rename(columns={'race_no_int': 'race_no', 'target_high_payout': 'label'}, inplace=True)
    oof_df.to_csv(MODEL_DIR / 'advanced_model_oof.csv', index=False)

    # Save all models
    for model_idx, fold_models in enumerate(models):
        for fold_idx, booster in enumerate(fold_models):
            model_file = MODEL_DIR / f'advanced_model_m{model_idx}_f{fold_idx}.txt'
            booster.save_model(str(model_file))

    # Save feature importance from first model
    first_booster = models[0][0]
    feature_importance = pd.DataFrame({
        'feature': first_booster.feature_name(),
        'importance': first_booster.feature_importance(importance_type='gain'),
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv(MODEL_DIR / 'advanced_model_feature_importance.csv', index=False)

    # Save metrics
    summary = {
        'payout_threshold': args.threshold,
        'n_folds': args.folds,
        'oof_roc_auc': metrics['oof_roc_auc'],
        'oof_average_precision': metrics['oof_average_precision'],
        'oof_classification_report': metrics['oof_classification_report'],
        'best_threshold_f1': best_threshold,
        'best_f1_score': best_f1,
        'precision_at_top_k': precision_at_k,
        'top_k': top_k,
        'ensemble_method': metrics['ensemble_method'],
        'model_weights': metrics['model_weights'],
        'individual_model_aucs': metrics['individual_model_aucs'],
        'n_features': len(feature_columns),
        'n_numeric_features': len(numeric_features),
        'n_categorical_features': len(categorical_features),
    }

    METRICS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"OOF ROC-AUC: {metrics['oof_roc_auc']:.4f}")
    print(f"OOF Avg Precision: {metrics['oof_average_precision']:.4f}")
    print(f"Best F1 Score: {best_f1:.4f} (threshold={best_threshold:.4f})")
    print(f"Precision@Top{top_k}: {precision_at_k:.4f}")
    print(f"Ensemble method: {metrics['ensemble_method']}")
    print(f"Individual model AUCs: {metrics['individual_model_aucs']}")
    print(f"\nMetrics saved to: {METRICS_PATH}")
    print(f"Models saved to: {MODEL_DIR}/advanced_model_m*_f*.txt")


if __name__ == '__main__':
    main()
