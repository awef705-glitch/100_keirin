#!/usr/bin/env python3
"""
Version 6 Refined: 重要特徴量のみを使用した最適化訓練

V6の新特徴量のうち、重要度が高いものだけを選択し、
V5のベストパラメータで訓練することで、過学習を防ぎつつ精度向上を目指す
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


def select_top_features(df, feature_importance_file, top_n=80):
    """
    特徴量重要度に基づいて上位N個の特徴量を選択

    V5とV6アンサンブルの両方の重要度を参照
    """
    print(f"\n  特徴量選択: 上位{top_n}個を使用")

    # V6の特徴量重要度を読み込み
    v6_fi = pd.read_csv(feature_importance_file)

    # V5の特徴量重要度も読み込み（参考）
    try:
        v5_fi = pd.read_csv('analysis/model_outputs/high_payout_model_v5_feature_importance.csv')
        # 両方の重要度の平均を取る
        merged_fi = v6_fi.merge(v5_fi, on='feature', how='outer', suffixes=('_v6', '_v5'))
        merged_fi['gain_v6'] = merged_fi['gain_v6'].fillna(0)
        merged_fi['gain_v5'] = merged_fi['gain_v5'].fillna(0)
        merged_fi['gain_avg'] = (merged_fi['gain_v6'] + merged_fi['gain_v5']) / 2
        merged_fi = merged_fi.sort_values('gain_avg', ascending=False)

        top_features = merged_fi.head(top_n)['feature'].tolist()
        print(f"  → V5とV6の重要度を統合して選択")

    except:
        # V5がなければV6だけ使う
        v6_fi = v6_fi.sort_values('gain', ascending=False)
        top_features = v6_fi.head(top_n)['feature'].tolist()
        print(f"  → V6の重要度のみで選択")

    # 利用可能な特徴量のみをフィルタ
    available_features = [f for f in top_features if f in df.columns]

    print(f"  → 選択された特徴量: {len(available_features)}個")

    return available_features


def main():
    print("=== Version 6 Refined 訓練開始 ===\n")

    # データ読み込み
    print("1. データ読み込み中...")
    df = pd.read_csv('data/training_dataset_ultra_v6.csv')
    print(f"   {len(df):,}行, {len(df.columns)}列")

    # カテゴリカル組み合わせのエンコーディング
    print("\n2. カテゴリカル組み合わせのエンコーディング中...")
    categorical_comb_cols = [c for c in df.columns if '_x_' in c and df[c].dtype == 'object']
    for col in categorical_comb_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 特徴量選択
    target_col = 'target_high_payout'
    exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col]

    all_feature_cols = [c for c in df.columns if c not in exclude_cols]

    # V6の特徴量重要度に基づいて上位80個を選択
    print("\n3. 特徴量選択中...")
    selected_features = select_top_features(
        df,
        'analysis/model_outputs/high_payout_model_v6_feature_importance.csv',
        top_n=80
    )

    X = df[selected_features]
    y = df[target_col]

    print(f"\n4. 訓練データ準備完了")
    print(f"   特徴量数: {len(selected_features)}")
    print(f"   Positive rate: {y.mean():.3f}")

    # V5のベストパラメータを使用
    print("\n5. V5ベストパラメータで訓練開始...")
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
    print(f"Precision@Top100: {oof_prec_k:.4f} ({oof_prec_k*100:.1f}%)")

    # 最終モデルを全データで訓練
    print("\n6. 最終モデルを全データで訓練中...")
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=int(np.mean([m['best_iteration'] for m in fold_metrics])),
        callbacks=[lgb.log_evaluation(50)]
    )

    # 保存
    print("\n7. 結果保存中...")
    model_file = 'analysis/model_outputs/high_payout_model_v6_refined.txt'
    final_model.save_model(model_file)
    print(f"   モデル保存: {model_file}")

    # メタデータ
    metadata = {
        'version': 'v6_refined',
        'n_features': len(selected_features),
        'n_samples': len(df),
        'positive_rate': float(y.mean()),
        'selected_features': selected_features,
        'params': best_params,
        'metrics': {
            'oof_roc_auc': float(oof_roc_auc),
            'oof_precision_at_top_k': float(oof_prec_k),
            'folds': fold_metrics
        }
    }

    metadata_file = 'analysis/model_outputs/high_payout_model_v6_refined_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"   メタデータ保存: {metadata_file}")

    # OOF予測
    oof_df = df[['race_date', 'track', target_col]].copy()
    oof_df['prediction'] = oof_predictions
    oof_file = 'analysis/model_outputs/high_payout_model_v6_refined_oof.csv'
    oof_df.to_csv(oof_file, index=False)
    print(f"   OOF予測保存: {oof_file}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'gain': final_model.feature_importance(importance_type='gain')
    }).sort_values('gain', ascending=False)

    fi_file = 'analysis/model_outputs/high_payout_model_v6_refined_feature_importance.csv'
    feature_importance.to_csv(fi_file, index=False)
    print(f"   特徴量重要度保存: {fi_file}")

    print("\n=== Top 20 重要特徴量 ===")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['gain']:.1f}")

    print("\n✅ Version 6 Refined 訓練完了！")

    # 比較表示
    print("\n" + "="*70)
    print("📊 モデル比較")
    print("="*70)
    print(f"{'バージョン':<20} {'特徴量数':<12} {'Precision@Top100':<20}")
    print("-"*70)
    print(f"{'V5 (ベスト)':<20} {84:<12} {67.0:<20.1f}%")
    print(f"{'V6 Ensemble':<20} {112:<12} {59.0:<20.1f}%")
    print(f"{'V6 Refined':<20} {len(selected_features):<12} {oof_prec_k*100:<20.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()
