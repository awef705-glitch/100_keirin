#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LightGBMハイパーパラメータチューニングスクリプト
Grid Searchで最適なパラメータを探索
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from pathlib import Path
import json
import sys

sys.path.append(str(Path.cwd()))
from analysis import prerace_model

def tune_hyperparameters():
    """ハイパーパラメータチューニング実行"""
    
    # キャッシュされたデータセットを読み込み
    print("Loading cached dataset...")
    dataset = prerace_model.load_cached_dataset()
    
    feature_cols = prerace_model._default_feature_columns()
    X = dataset[feature_cols]
    y = dataset['target_high_payout']
    
    print(f"Dataset: {len(dataset)} samples, {len(feature_cols)} features")
    print(f"Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # 探索するパラメータグリッド
    param_grid = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_data_in_leaf': [20, 30, 50],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0],
        'bagging_freq': [0, 5, 10],
    }
    
    # ベースモデル
    base_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    
    # 時系列分割（古いデータで訓練、新しいデータでテスト）
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("\nStarting Grid Search...")
    print(f"Parameter combinations: {len(param_grid['num_leaves']) * len(param_grid['learning_rate']) * len(param_grid['min_data_in_leaf']) * len(param_grid['feature_fraction']) * len(param_grid['bagging_fraction']) * len(param_grid['bagging_freq'])}")
    
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X, y)
    
    print("\n" + "="*80)
    print("Best Parameters:")
    print("="*80)
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest CV AUC: {grid_search.best_score_:.4f}")
    
    # 結果を保存
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']]
        }
    }
    
    output_path = Path('analysis/model_outputs/hyperparameter_tuning_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    
    # 最適パラメータで再訓練
    print("\nRetraining with best parameters...")
    best_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=200,
        random_state=42,
        **grid_search.best_params_
    )
    
    best_model.fit(X, y)
    
    # モデル保存
    model_path = Path('analysis/model_outputs/prerace_model_lgbm_tuned.txt')
    best_model.booster_.save_model(str(model_path))
    print(f"Tuned model saved to {model_path}")

if __name__ == '__main__':
    tune_hyperparameters()
