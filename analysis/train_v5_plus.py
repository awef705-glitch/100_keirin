#!/usr/bin/env python3
"""
V5 Plus: V5 + 厳選されたV6新特徴量で最終最適化
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


print('=== V5 Plus 最終訓練開始 ===\n')

# データ読み込み
print('1. データ読み込み中...')
df = pd.read_csv('data/training_dataset_v5_plus.csv')
print(f'   {len(df):,}行, {len(df.columns)}列')

# 特徴量選択
target_col = 'target_high_payout'
exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col]
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols]
y = df[target_col]

print(f'   特徴量数: {len(feature_cols)}')
print(f'   Positive rate: {y.mean():.3f}')

# V5ベストパラメータ
print('\n2. V5ベストパラメータで訓練...')
best_params = {
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

tscv = TimeSeriesSplit(n_splits=5)
oof_predictions = np.zeros(len(y))
fold_models = []
fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f'\n  Fold {fold_idx+1}/5 訓練中...')
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

    fold_models.append(model)

    print(f'    ROC-AUC: {roc_auc:.4f}, P@100: {prec_k:.2f}, Iter: {model.best_iteration}')

# OOF全体
oof_roc_auc = roc_auc_score(y, oof_predictions)
oof_prec_k = precision_at_k(y, oof_predictions, k=100)

print(f'\n=== OOF全体 ===')
print(f'ROC-AUC: {oof_roc_auc:.4f}')
print(f'Precision@Top100: {oof_prec_k:.4f} ({oof_prec_k*100:.1f}%)')

# 最終モデル
print('\n3. 最終モデル訓練中...')
train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=int(np.mean([m['best_iteration'] for m in fold_metrics])),
    callbacks=[lgb.log_evaluation(50)]
)

# 保存
model_file = 'analysis/model_outputs/high_payout_model_v5_plus.txt'
final_model.save_model(model_file)
print(f'   モデル保存: {model_file}')

metadata = {
    'version': 'v5_plus',
    'n_features': len(feature_cols),
    'n_samples': len(df),
    'positive_rate': float(y.mean()),
    'params': best_params,
    'metrics': {
        'oof_roc_auc': float(oof_roc_auc),
        'oof_precision_at_top_k': float(oof_prec_k),
        'folds': fold_metrics
    }
}

metadata_file = 'analysis/model_outputs/high_payout_model_v5_plus_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f'   メタデータ保存: {metadata_file}')

oof_df = df[['race_date', 'track', target_col]].copy()
oof_df['prediction'] = oof_predictions
oof_file = 'analysis/model_outputs/high_payout_model_v5_plus_oof.csv'
oof_df.to_csv(oof_file, index=False)
print(f'   OOF予測保存: {oof_file}')

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'gain': final_model.feature_importance(importance_type='gain')
}).sort_values('gain', ascending=False)

fi_file = 'analysis/model_outputs/high_payout_model_v5_plus_feature_importance.csv'
feature_importance.to_csv(fi_file, index=False)
print(f'   特徴量重要度保存: {fi_file}')

new_features = ['score_q25', 'score_concentration', 'category_x_month', 'score_dominance',
                'score_gap_top3_bottom', 'score_q75', 'sasiCnt_skew_proxy', 'score_kurtosis_proxy',
                'score_iqr', 'inner_outer_gap', 'inner_avg_score', 'outer_avg_score',
                'top3_inner_ratio', 'upgraded_count', 'downgraded_count', 'kyuhan_change_ratio',
                'makuriCnt_skew_proxy', 'nigeCnt_skew_proxy', 'backCnt_skew_proxy',
                'track_x_category', 'track_x_month']

print('\n=== Top 20 重要特徴量 ===')
for idx, row in feature_importance.head(20).iterrows():
    new_marker = ' 🆕' if row['feature'] in new_features else ''
    print(f'  {idx+1}. {row["feature"]}: {row["gain"]:.1f}{new_marker}')

print('\n' + '='*70)
print('📊 最終結果比較')
print('='*70)
print(f"{'バージョン':<20} {'特徴量数':<12} {'Precision@Top100':<20}")
print('-'*70)
print(f"{'V5':<20} {84:<12} {67.0:<20.1f}%")
print(f"{'V5 Plus':<20} {len(feature_cols):<12} {oof_prec_k*100:<20.1f}% ✅")
print('='*70)

print('\n✅ V5 Plus 訓練完了！')
