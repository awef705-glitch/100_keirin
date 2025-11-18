#!/usr/bin/env python3
"""
高度な技術の統合:
1. Permutation Importanceベースの特徴量選択
2. スタッキング (メタ学習)
3. ベイズ最適化
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import json


def precision_at_k(y_true, y_pred, k=100):
    if len(y_true) < k:
        k = len(y_true)
    top_k_idx = np.argsort(y_pred)[-k:]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, 'iloc') else y_true[top_k_idx].mean()


def select_features_by_permutation_importance(X, y, n_features=80, n_folds=5):
    """
    Permutation Importanceベースで特徴量選択

    Args:
        X: 特徴量DataFrame
        y: 目的変数
        n_features: 選択する特徴量数
        n_folds: CV分割数

    Returns:
        list: 選択された特徴量名のリスト
    """
    print(f"\n=== Permutation Importance 特徴量選択 ===")
    print(f"目標: 上位{n_features}特徴量を選択\n")

    # V5ベストパラメータでモデル訓練
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
    importances_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold_idx+1}/{n_folds}...")

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

        # Permutation Importance計算
        # LightGBMは直接サポートしていないので、validation setで計算
        perm_imp = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=10,
            random_state=42,
            scoring='roc_auc'
        )

        importances_list.append(perm_imp.importances_mean)

    # 平均重要度を計算
    mean_importances = np.mean(importances_list, axis=0)

    # DataFrameにまとめる
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_importances
    }).sort_values('importance', ascending=False)

    print(f"\n上位{min(20, len(importance_df))}特徴量:")
    for idx, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

    # 上位N特徴量を選択
    selected_features = importance_df.head(n_features)['feature'].tolist()

    print(f"\n✅ {len(selected_features)}特徴量を選択")

    return selected_features, importance_df


def train_stacking_model(X, y, n_folds=5):
    """
    スタッキング (メタ学習) モデルの訓練

    Base models: LightGBM (3種), CatBoost, XGBoost
    Meta model: LightGBM

    Args:
        X: 特徴量DataFrame
        y: 目的変数
        n_folds: CV分割数

    Returns:
        tuple: (meta_model, oof_predictions, metrics)
    """
    print("\n=== スタッキング訓練 ===")
    print("Base Models: LightGBM(3), CatBoost, XGBoost")
    print("Meta Model: LightGBM\n")

    tscv = TimeSeriesSplit(n_splits=n_folds)

    # Base modelsのOOF予測を保存
    base_oof_predictions = []
    base_model_names = []

    # Base Model 1: LightGBM (Balanced)
    print("1. LightGBM (Balanced)...")
    params_lgb1 = {
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
    oof_lgb1 = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params_lgb1,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )

        oof_lgb1[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    base_oof_predictions.append(oof_lgb1)
    base_model_names.append('lgb_balanced')
    print(f"   P@100={precision_at_k(y, oof_lgb1, k=100)*100:.1f}%")

    # Base Model 2: LightGBM (Conservative)
    print("2. LightGBM (Conservative)...")
    params_lgb2 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 43,
        'learning_rate': 0.02,
        'num_leaves': 27,
        'max_depth': 7,
        'min_child_samples': 40,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': 2.5,
    }
    oof_lgb2 = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params_lgb2,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )

        oof_lgb2[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    base_oof_predictions.append(oof_lgb2)
    base_model_names.append('lgb_conservative')
    print(f"   P@100={precision_at_k(y, oof_lgb2, k=100)*100:.1f}%")

    # Base Model 3: LightGBM (Aggressive)
    print("3. LightGBM (Aggressive)...")
    params_lgb3 = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 44,
        'learning_rate': 0.04,
        'num_leaves': 63,
        'max_depth': 10,
        'min_child_samples': 20,
        'scale_pos_weight': 2.5,
    }
    oof_lgb3 = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params_lgb3,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
        )

        oof_lgb3[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    base_oof_predictions.append(oof_lgb3)
    base_model_names.append('lgb_aggressive')
    print(f"   P@100={precision_at_k(y, oof_lgb3, k=100)*100:.1f}%")

    # Base Model 4: CatBoost
    print("4. CatBoost...")
    try:
        from catboost import CatBoostClassifier

        oof_cat = np.zeros(len(y))
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.03,
                depth=8,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=False,
                early_stopping_rounds=30,
                scale_pos_weight=2.5,
            )

            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            oof_cat[val_idx] = model.predict_proba(X_val)[:, 1]

        base_oof_predictions.append(oof_cat)
        base_model_names.append('catboost')
        print(f"   P@100={precision_at_k(y, oof_cat, k=100)*100:.1f}%")
    except ImportError:
        print("   CatBoostがインストールされていません - スキップ")

    # Base Model 5: XGBoost
    print("5. XGBoost...")
    try:
        import xgboost as xgb

        oof_xgb = np.zeros(len(y))
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                {
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
                },
                dtrain,
                num_boost_round=300,
                evals=[(dval, 'val')],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            oof_xgb[val_idx] = model.predict(dval, iteration_range=(0, model.best_iteration))

        base_oof_predictions.append(oof_xgb)
        base_model_names.append('xgboost')
        print(f"   P@100={precision_at_k(y, oof_xgb, k=100)*100:.1f}%")
    except ImportError:
        print("   XGBoostがインストールされていません - スキップ")

    # Meta modelの訓練
    print("\n6. Meta Model (LightGBM)...")
    meta_features = np.column_stack(base_oof_predictions)
    meta_feature_df = pd.DataFrame(meta_features, columns=base_model_names)

    meta_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'seed': 42,
        'learning_rate': 0.01,
        'num_leaves': 15,
        'max_depth': 5,
        'min_child_samples': 50,
        'scale_pos_weight': 2.5,
    }

    meta_oof = np.zeros(len(y))
    for train_idx, val_idx in tscv.split(meta_feature_df):
        X_meta_train = meta_feature_df.iloc[train_idx]
        X_meta_val = meta_feature_df.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        train_data = lgb.Dataset(X_meta_train, label=y_train)
        val_data = lgb.Dataset(X_meta_val, label=y_val, reference=train_data)

        meta_model = lgb.train(
            meta_params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
        )

        meta_oof[val_idx] = meta_model.predict(X_meta_val, num_iteration=meta_model.best_iteration)

    meta_roc = roc_auc_score(y, meta_oof)
    meta_prec = precision_at_k(y, meta_oof, k=100)

    print(f"\n=== スタッキング結果 ===")
    print(f"Meta Model: ROC-AUC={meta_roc:.4f}, P@100={meta_prec*100:.1f}%")

    # 個別モデルとの比較
    print(f"\n=== Base Models比較 ===")
    for idx, name in enumerate(base_model_names):
        prec = precision_at_k(y, base_oof_predictions[idx], k=100)
        print(f"  {name}: P@100={prec*100:.1f}%")

    return meta_model, meta_oof, {
        'meta_roc_auc': meta_roc,
        'meta_precision_k': meta_prec,
        'base_models': base_model_names
    }


def main():
    print("=== 高度技術の適用 ===\n")

    # V8データ読み込み
    print("1. V8データ読み込み中...")
    v8_data = pd.read_csv('data/training_dataset_v8_ultimate.csv')
    print(f"   {len(v8_data):,}行, {len(v8_data.columns)}列")

    target_col = 'target_high_payout'
    exclude_cols = ['category', 'grade', 'keirin_cd', 'race_date', 'track', target_col, 'race_no', 'race_no_str', 'race_no_int']
    feature_cols = [c for c in v8_data.columns if c not in exclude_cols and v8_data[c].dtype in [np.float64, np.int64]]

    X = v8_data[feature_cols]
    y = v8_data[target_col]

    print(f"   特徴量数: {len(feature_cols)}")
    print(f"   Positive rate: {y.mean():.3f}")

    # 1. Permutation Importanceベース特徴量選択
    selected_features, importance_df = select_features_by_permutation_importance(
        X, y, n_features=80, n_folds=5
    )

    # 選択された特徴量で再訓練
    X_selected = X[selected_features]

    # 2. スタッキング
    meta_model, stacking_oof, stacking_metrics = train_stacking_model(X_selected, y, n_folds=5)

    # 結果保存
    print("\n3. 結果保存中...")

    # 特徴量重要度
    importance_df.to_csv('analysis/model_outputs/v8_permutation_importance.csv', index=False)
    print("   特徴量重要度保存: v8_permutation_importance.csv")

    # スタッキングOOF
    oof_df = v8_data[['race_date', 'track', target_col]].copy()
    oof_df['stacking_prediction'] = stacking_oof
    oof_df.to_csv('analysis/model_outputs/v8_stacking_oof.csv', index=False)
    print("   スタッキングOOF保存: v8_stacking_oof.csv")

    # メタデータ
    metadata = {
        'version': 'v8_advanced',
        'n_features_original': len(feature_cols),
        'n_features_selected': len(selected_features),
        'selected_features': selected_features,
        'stacking_metrics': stacking_metrics,
    }

    with open('analysis/model_outputs/v8_advanced_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("   メタデータ保存: v8_advanced_metadata.json")

    print("\n✅ 高度技術の適用完了！")


if __name__ == '__main__':
    main()
