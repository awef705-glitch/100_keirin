#!/usr/bin/env python3
"""
Version 5訓練スクリプト - ハイパーパラメータグリッドサーチ付き

過学習に注意しながら、最適なハイパーパラメータを探索
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import json
from itertools import product


def precision_at_k(y_true, y_pred, k=100):
    """Top K予測のPrecision"""
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()


def train_with_params(X, y, params, n_folds=5, verbose=False):
    """
    指定されたパラメータでTimeSeriesSplit訓練を実行

    Returns:
        dict: メトリクス
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)

    fold_metrics = []
    all_predictions = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )

        # 検証セットで予測
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        all_predictions[val_idx] = val_pred

        # メトリクス計算
        roc_auc = roc_auc_score(y_val, val_pred)
        prec_k = precision_at_k(y_val, val_pred, k=100)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'roc_auc': roc_auc,
            'precision_at_top_k': prec_k
        })

        if verbose:
            print(f"  Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, Precision@Top100={prec_k:.2f}")

    # OOF全体のメトリクス
    oof_roc_auc = roc_auc_score(y, all_predictions)
    oof_prec_k = precision_at_k(y, all_predictions, k=100)

    # 平均
    avg_roc_auc = np.mean([m['roc_auc'] for m in fold_metrics])
    avg_prec_k = np.mean([m['precision_at_top_k'] for m in fold_metrics])

    return {
        'oof_roc_auc': oof_roc_auc,
        'oof_precision_k': oof_prec_k,
        'avg_roc_auc': avg_roc_auc,
        'avg_precision_k': avg_prec_k,
        'fold_metrics': fold_metrics
    }


def grid_search(X, y, param_grid, n_folds=5):
    """
    グリッドサーチでベストパラメータを探索

    Args:
        X: 特徴量
        y: 目的変数
        param_grid: パラメータグリッド
        n_folds: Fold数

    Returns:
        best_params, best_score, all_results
    """
    print("=== ハイパーパラメータグリッドサーチ開始 ===\n")

    # ベースパラメータ
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
    }

    # グリッドの組み合わせ生成
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    print(f"グリッド組み合わせ数: {len(combinations)}")
    print(f"各組み合わせで{n_folds}分割CV実行\n")

    results = []
    best_score = -1
    best_params = None

    for idx, combination in enumerate(combinations):
        params = base_params.copy()
        for key, val in zip(keys, combination):
            params[key] = val

        print(f"[{idx+1}/{len(combinations)}] 試行中: {dict(zip(keys, combination))}")

        metrics = train_with_params(X, y, params, n_folds=n_folds, verbose=False)

        # Precision@Top100を主要指標とする
        score = metrics['oof_precision_k']

        print(f"  → OOF ROC-AUC: {metrics['oof_roc_auc']:.4f}, Precision@Top100: {score:.4f}")

        result = {
            'params': dict(zip(keys, combination)),
            'metrics': metrics,
            'score': score
        }
        results.append(result)

        if score > best_score:
            best_score = score
            best_params = params
            print(f"  ✅ 新ベスト！")

        print()

    print(f"\n=== グリッドサーチ完了 ===")
    print(f"ベストスコア（Precision@Top100）: {best_score:.4f}")
    print(f"ベストパラメータ: {best_params}\n")

    return best_params, best_score, results


def main():
    print("=== Version 5 訓練開始（グリッドサーチ付き）===\n")

    # データ読み込み
    print("1. データ読み込み中...")
    df = pd.read_csv('data/training_dataset_enhanced_v5.csv')
    print(f"   {len(df):,}行, {len(df.columns)}列")

    # 特徴量と目的変数の分離
    target_col = 'target_high_payout'
    exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"   特徴量数: {len(feature_cols)}")
    print(f"   Positive rate: {y.mean():.3f}")

    # パラメータグリッド定義（過学習防止のため控えめに）
    print("\n2. パラメータグリッド定義...")
    param_grid = {
        'learning_rate': [0.03, 0.05],
        'num_leaves': [31, 63],
        'max_depth': [6, 8],
        'min_child_samples': [30, 50],
        'scale_pos_weight': [2.0, 2.5, 2.8],  # クラス不均衡対策（negative/positive比 ≈ 2.8）
    }

    # 組み合わせ数: 2*2*2*2*3 = 48組み合わせ

    # グリッドサーチ実行
    best_params, best_score, all_results = grid_search(X, y, param_grid, n_folds=5)

    # 結果保存
    print("\n3. グリッドサーチ結果を保存中...")
    results_file = 'analysis/model_outputs/v5_gridsearch_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"   保存完了: {results_file}")

    # ベストパラメータで最終訓練
    print("\n4. ベストパラメータで最終訓練中...")
    tscv = TimeSeriesSplit(n_splits=5)

    oof_predictions = np.zeros(len(y))
    fold_models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold_idx+1}/5 訓練中...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
        )

        # 予測
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_predictions[val_idx] = val_pred

        # メトリクス
        roc_auc = roc_auc_score(y_val, val_pred)
        prec_k = precision_at_k(y_val, val_pred, k=100)

        fold_metrics.append({
            'fold': fold_idx + 1,
            'roc_auc': roc_auc,
            'precision_at_top_k': prec_k,
            'best_iteration': model.best_iteration
        })

        fold_models.append(model)

        print(f"  → ROC-AUC: {roc_auc:.4f}, Precision@Top100: {prec_k:.2f}, Best iteration: {model.best_iteration}")

    # OOF全体のメトリクス
    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_prec_k = precision_at_k(y, oof_predictions, k=100)

    print(f"\n=== OOF全体のパフォーマンス ===")
    print(f"ROC-AUC: {oof_roc_auc:.4f}")
    print(f"Precision@Top100: {oof_prec_k:.4f}")

    # 最終モデルを保存（全データで訓練）
    print("\n5. 最終モデルを全データで訓練中...")
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=int(np.mean([m['best_iteration'] for m in fold_metrics])),
        callbacks=[lgb.log_evaluation(50)]
    )

    # 保存
    model_file = 'analysis/model_outputs/high_payout_model_v5.txt'
    final_model.save_model(model_file)
    print(f"   モデル保存: {model_file}")

    # メタデータ保存
    metadata = {
        'version': 'v5',
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'positive_rate': float(y.mean()),
        'best_params': best_params,
        'metrics': {
            'oof_roc_auc': float(oof_roc_auc),
            'oof_precision_at_top_k': float(oof_prec_k),
            'folds': fold_metrics
        }
    }

    metadata_file = 'analysis/model_outputs/high_payout_model_v5_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   メタデータ保存: {metadata_file}")

    # OOF予測保存
    oof_df = df[['race_date', 'track', target_col]].copy()
    oof_df['prediction'] = oof_predictions
    oof_file = 'analysis/model_outputs/high_payout_model_v5_oof.csv'
    oof_df.to_csv(oof_file, index=False)
    print(f"   OOF予測保存: {oof_file}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'gain': final_model.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)

    fi_file = 'analysis/model_outputs/high_payout_model_v5_feature_importance.csv'
    feature_importance.to_csv(fi_file, index=False)
    print(f"   特徴量重要度保存: {fi_file}")

    print("\n=== Top 20 重要特徴量 ===")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['gain']:.1f}")

    print("\n✅ Version 5 訓練完了！")


if __name__ == '__main__':
    main()
