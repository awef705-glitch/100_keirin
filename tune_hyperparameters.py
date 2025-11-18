#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ハイパーパラメータチューニング（Optuna）
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score


def objective(trial, X, y):
    """
    Optunaの目的関数
    """
    # ハイパーパラメータの探索空間
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,

        # 探索するハイパーパラメータ
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 15.0),
        'scale_pos_weight': 2.68,  # Fixed based on class imbalance
    }

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        preds = model.predict(X_valid)
        score = roc_auc_score(y_valid, preds)
        cv_scores.append(score)

    return np.mean(cv_scores)


def main():
    print("=" * 80)
    print("ハイパーパラメータチューニング（Optuna）")
    print("=" * 80)

    # Load data
    input_file = Path('data/clean_training_dataset.csv')
    df = pd.read_csv(input_file)

    print(f"\n📂 Loaded: {len(df):,} races")

    # Prepare features
    target_col = 'target_high_payout'
    exclude_cols = [target_col, 'race_date', 'track', 'keirin_cd', 'grade', 'category']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    print(f"📋 Features: {len(feature_cols)}")
    print(f"🎯 Target: {target_col}")
    print(f"⚖️  Class balance: {(y==0).sum()}/{(y==1).sum()} (neg/pos)")

    # Create Optuna study
    print(f"\n🔍 Starting hyperparameter search...")
    print(f"   Trials: 100")
    print(f"   Timeout: None")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=50,  # 50 trials for good optimization (was 100)
        show_progress_bar=True,
        n_jobs=1,  # Sequential for stability
    )

    print(f"\n" + "=" * 80)
    print("✅ Optimization Complete!")
    print("=" * 80)

    print(f"\n🏆 Best ROC-AUC: {study.best_value:.4f}")
    print(f"\n📊 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # Save best parameters
    output_dir = Path('analysis/model_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = output_dir / 'best_hyperparameters.json'
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)

    print(f"\n💾 Best parameters saved to: {best_params_path}")

    # Save study for later analysis
    study_path = output_dir / 'optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    print(f"💾 Study saved to: {study_path}")

    print(f"\n" + "=" * 80)
    print("Next step: Re-train model with best parameters")
    print("Run: python3 train_clean_model.py --use-tuned-params")
    print("=" * 80)


if __name__ == "__main__":
    main()
