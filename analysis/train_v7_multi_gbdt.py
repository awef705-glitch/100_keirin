#!/usr/bin/env python3
"""
Version 7: 選手個別データ + マルチGBDT（LightGBM, CatBoost, XGBoost）

3つのGBDTライブラリで訓練し、最高精度を追求
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import json


def precision_at_k(y_true, y_pred, k=100):
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()


def train_lightgbm(X, y, n_folds=5):
    """LightGBM訓練"""
    print("\n=== LightGBM 訓練 ===")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 8,
        'min_child_samples': 30,
        'scale_pos_weight': 2.5,
    }

    tscv = TimeSeriesSplit(n_splits=n_folds)
    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

        roc_auc = roc_auc_score(y_val, val_pred)
        prec_k = precision_at_k(y_val, val_pred, k=100)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'roc_auc': roc_auc,
            'precision_at_top_k': prec_k,
            'best_iteration': model.best_iteration
        })

        print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, P@100={prec_k:.2f}")

    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_prec_k = precision_at_k(y, oof_predictions, k=100)

    print(f"  → LightGBM OOF: ROC-AUC={oof_roc_auc:.4f}, P@100={oof_prec_k:.4f}")

    return oof_predictions, fold_metrics, oof_roc_auc, oof_prec_k


def train_catboost(X, y, n_folds=5):
    """CatBoost訓練"""
    print("\n=== CatBoost 訓練 ===")

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  ⚠️  CatBoostがインストールされていません")
        return None, None, 0, 0

    params = {
        'iterations': 300,
        'learning_rate': 0.03,
        'depth': 8,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 30,
        'scale_pos_weight': 2.5,
    }

    tscv = TimeSeriesSplit(n_splits=n_folds)
    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        val_pred = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = val_pred

        roc_auc = roc_auc_score(y_val, val_pred)
        prec_k = precision_at_k(y_val, val_pred, k=100)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'roc_auc': roc_auc,
            'precision_at_top_k': prec_k,
        })

        print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, P@100={prec_k:.2f}")

    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_prec_k = precision_at_k(y, oof_predictions, k=100)

    print(f"  → CatBoost OOF: ROC-AUC={oof_roc_auc:.4f}, P@100={oof_prec_k:.4f}")

    return oof_predictions, fold_metrics, oof_roc_auc, oof_prec_k


def train_xgboost(X, y, n_folds=5):
    """XGBoost訓練"""
    print("\n=== XGBoost 訓練 ===")

    try:
        import xgboost as xgb
    except ImportError:
        print("  ⚠️  XGBoostがインストールされていません")
        return None, None, 0, 0

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.03,
        'max_depth': 8,
        'min_child_weight': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 2.5,
        'seed': 42,
        'verbosity': 0,
    }

    tscv = TimeSeriesSplit(n_splits=n_folds)
    oof_predictions = np.zeros(len(y))
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dval, 'val')],
            early_stopping_rounds=30,
            verbose_eval=False
        )

        val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
        oof_predictions[val_idx] = val_pred

        roc_auc = roc_auc_score(y_val, val_pred)
        prec_k = precision_at_k(y_val, val_pred, k=100)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'roc_auc': roc_auc,
            'precision_at_top_k': prec_k,
            'best_iteration': model.best_iteration
        })

        print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, P@100={prec_k:.2f}")

    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_prec_k = precision_at_k(y, oof_predictions, k=100)

    print(f"  → XGBoost OOF: ROC-AUC={oof_roc_auc:.4f}, P@100={oof_prec_k:.4f}")

    return oof_predictions, fold_metrics, oof_roc_auc, oof_prec_k


def main():
    print("=== Version 7 マルチGBDT訓練開始 ===\n")

    # データ読み込み
    print("1. データ読み込み中...")
    df = pd.read_csv('data/training_dataset_v7_individual.csv')
    print(f"   {len(df):,}行, {len(df.columns)}列")

    # 特徴量選択
    target_col = 'target_high_payout'
    exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"   特徴量数: {len(feature_cols)}")
    print(f"   Positive rate: {y.mean():.3f}")

    # 3つのGBDTで訓練
    print("\n2. マルチGBDT訓練開始...")

    results = {}

    # LightGBM
    lgb_oof, lgb_metrics, lgb_roc, lgb_prec = train_lightgbm(X, y)
    results['lightgbm'] = {
        'oof': lgb_oof,
        'metrics': lgb_metrics,
        'oof_roc_auc': lgb_roc,
        'oof_precision_k': lgb_prec
    }

    # CatBoost
    cat_oof, cat_metrics, cat_roc, cat_prec = train_catboost(X, y)
    if cat_oof is not None:
        results['catboost'] = {
            'oof': cat_oof,
            'metrics': cat_metrics,
            'oof_roc_auc': cat_roc,
            'oof_precision_k': cat_prec
        }

    # XGBoost
    xgb_oof, xgb_metrics, xgb_roc, xgb_prec = train_xgboost(X, y)
    if xgb_oof is not None:
        results['xgboost'] = {
            'oof': xgb_oof,
            'metrics': xgb_metrics,
            'oof_roc_auc': xgb_roc,
            'oof_precision_k': xgb_prec
        }

    # アンサンブル（利用可能なモデルの平均）
    print("\n=== アンサンブル予測 ===")
    available_oofs = [r['oof'] for r in results.values() if r['oof'] is not None]

    if len(available_oofs) > 1:
        ensemble_oof = np.mean(available_oofs, axis=0)
        ensemble_roc = roc_auc_score(y, ensemble_oof)
        ensemble_prec = precision_at_k(y, ensemble_oof, k=100)
        print(f"  Ensemble ({len(available_oofs)}モデル): ROC-AUC={ensemble_roc:.4f}, P@100={ensemble_prec:.4f}")

        results['ensemble'] = {
            'oof': ensemble_oof,
            'oof_roc_auc': ensemble_roc,
            'oof_precision_k': ensemble_prec
        }

    # 結果比較
    print("\n" + "="*70)
    print("📊 モデル比較")
    print("="*70)
    print(f"{'モデル':<20} {'ROC-AUC':<15} {'Precision@Top100':<20}")
    print("-"*70)

    for model_name, result in results.items():
        if result['oof'] is not None:
            prec_pct = result['oof_precision_k'] * 100
            print(f"{model_name:<20} {result['oof_roc_auc']:<15.4f} {prec_pct:<20.1f}%")

    print("="*70)

    # 最良モデルの特定
    best_model = max(results.items(), key=lambda x: x[1]['oof_precision_k'])
    print(f"\n🏆 最良モデル: {best_model[0]} (P@100={best_model[1]['oof_precision_k']*100:.1f}%)")

    # 保存
    print("\n3. 結果保存中...")

    # OOF予測
    oof_df = df[['race_date', 'track', target_col]].copy()
    for model_name, result in results.items():
        if result['oof'] is not None:
            oof_df[f'{model_name}_prediction'] = result['oof']

    oof_file = 'analysis/model_outputs/high_payout_model_v7_oof.csv'
    oof_df.to_csv(oof_file, index=False)
    print(f"   OOF予測保存: {oof_file}")

    # メタデータ
    metadata = {
        'version': 'v7_multi_gbdt',
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'positive_rate': float(y.mean()),
        'models': {}
    }

    for model_name, result in results.items():
        if result['oof'] is not None:
            metadata['models'][model_name] = {
                'oof_roc_auc': float(result['oof_roc_auc']),
                'oof_precision_at_top_k': float(result['oof_precision_k']),
            }
            if 'metrics' in result:
                metadata['models'][model_name]['folds'] = result['metrics']

    metadata_file = 'analysis/model_outputs/high_payout_model_v7_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   メタデータ保存: {metadata_file}")

    print("\n✅ Version 7 マルチGBDT訓練完了！")

    # V5との比較
    print("\n" + "="*70)
    print("📊 V5 vs V7 比較")
    print("="*70)
    print(f"{'バージョン':<20} {'特徴量数':<12} {'Precision@Top100':<20}")
    print("-"*70)
    print(f"{'V5 (ベスト)':<20} {84:<12} {67.0:<20.1f}%")
    print(f"{'V7 (LightGBM)':<20} {len(feature_cols):<12} {lgb_prec*100:<20.1f}%")
    if cat_oof is not None:
        print(f"{'V7 (CatBoost)':<20} {len(feature_cols):<12} {cat_prec*100:<20.1f}%")
    if xgb_oof is not None:
        print(f"{'V7 (XGBoost)':<20} {len(feature_cols):<12} {xgb_prec*100:<20.1f}%")
    if 'ensemble' in results:
        print(f"{'V7 (Ensemble)':<20} {len(feature_cols):<12} {results['ensemble']['oof_precision_k']*100:<20.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()
