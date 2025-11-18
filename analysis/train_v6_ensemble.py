#!/usr/bin/env python3
"""
Version 6 アンサンブル学習訓練スクリプト

複数の異なるハイパーパラメータのモデルを訓練し、
予測を統合することでロバスト性と精度を向上
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json


def precision_at_k(y_true, y_pred, k=100):
    """Top K予測のPrecision"""
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()


def train_single_model(X, y, params, n_folds=5, model_name="model"):
    """
    単一モデルの訓練とOOF予測の生成

    Returns:
        oof_predictions, fold_models, fold_metrics
    """
    print(f"\n  === {model_name} 訓練中 ===")

    tscv = TimeSeriesSplit(n_splits=n_folds)

    oof_predictions = np.zeros(len(y))
    fold_models = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 訓練
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
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

        print(f"    Fold {fold_idx+1}: ROC-AUC={roc_auc:.4f}, P@100={prec_k:.2f}, Iter={model.best_iteration}")

    # OOF全体メトリクス
    oof_roc_auc = roc_auc_score(y, oof_predictions)
    oof_prec_k = precision_at_k(y, oof_predictions, k=100)

    print(f"  → {model_name} OOF: ROC-AUC={oof_roc_auc:.4f}, P@100={oof_prec_k:.4f}")

    return oof_predictions, fold_models, fold_metrics


def train_ensemble(X, y, n_folds=5):
    """
    アンサンブル学習：複数モデルを訓練して予測を統合

    Returns:
        ensemble_oof, all_models_info
    """
    print("\n=== アンサンブル学習開始 ===")

    # V5のベストパラメータをベースに、3つの異なるモデルを作成
    base_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
    }

    # モデル1: V5ベスト（バランス型）
    params1 = base_params.copy()
    params1.update({
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 8,
        'min_child_samples': 30,
        'scale_pos_weight': 2.5,
    })

    # モデル2: より保守的（過学習防止）
    params2 = base_params.copy()
    params2.update({
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 50,
        'scale_pos_weight': 2.5,
        'feature_fraction': 0.8,  # ランダム特徴量選択
        'bagging_fraction': 0.8,  # ランダムサンプリング
        'bagging_freq': 5,
    })

    # モデル3: より複雑（高表現力）
    params3 = base_params.copy()
    params3.update({
        'learning_rate': 0.04,
        'num_leaves': 63,
        'max_depth': 10,
        'min_child_samples': 20,
        'scale_pos_weight': 2.5,
        'lambda_l1': 0.1,  # L1正則化
        'lambda_l2': 0.1,  # L2正則化
    })

    # 各モデルを訓練
    models_info = []

    oof1, models1, metrics1 = train_single_model(X, y, params1, n_folds, "Model1_Balanced")
    models_info.append({'name': 'Model1_Balanced', 'params': params1, 'oof': oof1, 'models': models1, 'metrics': metrics1})

    oof2, models2, metrics2 = train_single_model(X, y, params2, n_folds, "Model2_Conservative")
    models_info.append({'name': 'Model2_Conservative', 'params': params2, 'oof': oof2, 'models': models2, 'metrics': metrics2})

    oof3, models3, metrics3 = train_single_model(X, y, params3, n_folds, "Model3_Complex")
    models_info.append({'name': 'Model3_Complex', 'params': params3, 'oof': oof3, 'models': models3, 'metrics': metrics3})

    # アンサンブル予測（単純平均）
    print("\n  === アンサンブル予測の統合 ===")
    ensemble_oof = (oof1 + oof2 + oof3) / 3

    # アンサンブルのメトリクス
    ensemble_roc_auc = roc_auc_score(y, ensemble_oof)
    ensemble_prec_k = precision_at_k(y, ensemble_oof, k=100)

    print(f"  → Ensemble OOF: ROC-AUC={ensemble_roc_auc:.4f}, P@100={ensemble_prec_k:.4f}")

    # 各モデルとアンサンブルの比較
    print("\n  === モデル比較 ===")
    for info in models_info:
        oof_roc = roc_auc_score(y, info['oof'])
        oof_prec = precision_at_k(y, info['oof'], k=100)
        print(f"    {info['name']}: ROC-AUC={oof_roc:.4f}, P@100={oof_prec:.4f}")

    print(f"    Ensemble:      ROC-AUC={ensemble_roc_auc:.4f}, P@100={ensemble_prec_k:.4f} ✅")

    return ensemble_oof, models_info


def main():
    print("=== Version 6 アンサンブル訓練開始 ===\n")

    # データ読み込み
    print("1. データ読み込み中...")
    df = pd.read_csv('data/training_dataset_ultra_v6.csv')
    print(f"   {len(df):,}行, {len(df.columns)}列")

    # 特徴量と目的変数の分離
    target_col = 'target_high_payout'
    exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col]

    # カテゴリカル組み合わせカラムの処理
    categorical_comb_cols = [c for c in df.columns if '_x_' in c and df[c].dtype == 'object']

    # Label Encodingでカテゴリカル組み合わせを数値化
    print("\n2. カテゴリカル組み合わせのエンコーディング中...")
    for col in categorical_comb_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"   {col}: {df[col].nunique()}ユニーク値")

    # 特徴量選択
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    print(f"\n3. 訓練データ準備完了")
    print(f"   特徴量数: {len(feature_cols)}")
    print(f"   Positive rate: {y.mean():.3f}")

    # アンサンブル訓練
    print("\n4. アンサンブル訓練開始...")
    ensemble_oof, models_info = train_ensemble(X, y, n_folds=5)

    # 結果保存
    print("\n5. 結果保存中...")

    # OOF予測保存
    oof_df = df[['race_date', 'track', target_col]].copy()
    oof_df['ensemble_prediction'] = ensemble_oof

    # 各モデルの予測も保存
    for info in models_info:
        oof_df[f'{info["name"]}_prediction'] = info['oof']

    oof_file = 'analysis/model_outputs/high_payout_model_v6_ensemble_oof.csv'
    oof_df.to_csv(oof_file, index=False)
    print(f"   OOF予測保存: {oof_file}")

    # メタデータ保存
    ensemble_roc_auc = roc_auc_score(y, ensemble_oof)
    ensemble_prec_k = precision_at_k(y, ensemble_oof, k=100)

    metadata = {
        'version': 'v6_ensemble',
        'n_features': len(feature_cols),
        'n_samples': len(df),
        'positive_rate': float(y.mean()),
        'ensemble_metrics': {
            'oof_roc_auc': float(ensemble_roc_auc),
            'oof_precision_at_top_k': float(ensemble_prec_k),
        },
        'models': []
    }

    for info in models_info:
        model_oof_roc = roc_auc_score(y, info['oof'])
        model_oof_prec = precision_at_k(y, info['oof'], k=100)

        metadata['models'].append({
            'name': info['name'],
            'params': info['params'],
            'oof_roc_auc': float(model_oof_roc),
            'oof_precision_at_top_k': float(model_oof_prec),
            'folds': info['metrics']
        })

    metadata_file = 'analysis/model_outputs/high_payout_model_v6_ensemble_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   メタデータ保存: {metadata_file}")

    # 各モデルを保存（Fold 0のモデル）
    for info in models_info:
        model_file = f'analysis/model_outputs/high_payout_model_v6_{info["name"]}.txt'
        info['models'][0].save_model(model_file)
        print(f"   {info['name']} モデル保存: {model_file}")

    # 特徴量重要度（Model1から取得）
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'gain': models_info[0]['models'][0].feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)

    fi_file = 'analysis/model_outputs/high_payout_model_v6_feature_importance.csv'
    feature_importance.to_csv(fi_file, index=False)
    print(f"   特徴量重要度保存: {fi_file}")

    print("\n=== Top 20 重要特徴量 ===")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['gain']:.1f}")

    print("\n✅ Version 6 アンサンブル訓練完了！")

    # 最終結果サマリー
    print("\n" + "="*60)
    print("📊 V6 アンサンブル 最終結果")
    print("="*60)
    print(f"特徴量数: {len(feature_cols)}")
    print(f"レース数: {len(df):,}")
    print(f"\nアンサンブル性能:")
    print(f"  ROC-AUC (OOF):        {ensemble_roc_auc:.4f}")
    print(f"  Precision@Top100:     {ensemble_prec_k:.4f} ({ensemble_prec_k*100:.1f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
