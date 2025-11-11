#!/usr/bin/env python3
"""
改善版予測モデル - 精度80%以上を目指す
"""
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score


def get_player_features(player_name: str, player_stats: dict, track: str = None,
                        grade: str = None, category: str = None) -> dict:
    """選手の詳細な特徴量を取得"""
    if player_name not in player_stats:
        # 未知の選手はデフォルト値
        return {
            "win_rate": 0.1,
            "place_2_rate": 0.1,
            "place_3_rate": 0.1,
            "top3_rate": 0.3,
            "avg_payout": 5000,
            "high_payout_rate": 0.2,
            "races": 0.0,
            "recent_win_rate": 0.1,
            "recent_top3_rate": 0.3,
            "track_win_rate": 0.1,
            "grade_win_rate": 0.1,
            "category_win_rate": 0.1,
            "consistency": 0.0,
        }

    stats = player_stats[player_name]

    # 基本統計
    features = {
        "win_rate": stats["win_rate"],
        "place_2_rate": stats["place_2_rate"],
        "place_3_rate": stats["place_3_rate"],
        "top3_rate": stats["top3_rate"],
        "avg_payout": stats["avg_payout"],
        "high_payout_rate": stats["high_payout_rate"],
        "races": min(stats["races"], 500) / 500,
    }

    # 最近のパフォーマンス
    features["recent_win_rate"] = stats.get("recent_win_rate", stats["win_rate"])
    features["recent_top3_rate"] = stats.get("recent_top3_rate", stats["top3_rate"])

    # 場所別勝率
    if track and track in stats.get("by_track", {}):
        features["track_win_rate"] = stats["by_track"][track]["win_rate"]
    else:
        features["track_win_rate"] = stats["win_rate"]

    # グレード別勝率
    if grade and grade in stats.get("by_grade", {}):
        features["grade_win_rate"] = stats["by_grade"][grade]["win_rate"]
    else:
        features["grade_win_rate"] = stats["win_rate"]

    # カテゴリー別勝率
    if category and category in stats.get("by_category", {}):
        features["category_win_rate"] = stats["by_category"][category]["win_rate"]
    else:
        features["category_win_rate"] = stats["win_rate"]

    # 安定性（最近の成績と全体の成績の差）
    features["consistency"] = 1.0 - abs(features["recent_win_rate"] - stats["win_rate"])

    return features


def build_features(df: pd.DataFrame, player_stats: dict) -> tuple:
    """改善された特徴量を構築"""
    print("\n特徴量を構築中...")

    X_list = []
    y_list = []

    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"  処理中: {idx}/{len(df)} レース")

        # ラベル
        trifecta_payout = row.get("trifecta_payout", "0円")
        try:
            payout = int(str(trifecta_payout).replace("円", "").replace(",", ""))
        except:
            payout = 0

        y = 1 if payout >= 10000 else 0

        # レース情報
        track = row.get("track", "不明")
        grade = row.get("grade", "不明")
        category = row.get("category", "不明")

        # 選手名
        pos1_name = row.get("pos1_name")
        pos2_name = row.get("pos2_name")
        pos3_name = row.get("pos3_name")

        if pd.isna(pos1_name) or pd.isna(pos2_name) or pd.isna(pos3_name):
            continue

        # 選手統計を取得（場所・グレード・カテゴリー考慮）
        pos1_stats = get_player_features(pos1_name, player_stats, track, grade, category)
        pos2_stats = get_player_features(pos2_name, player_stats, track, grade, category)
        pos3_stats = get_player_features(pos3_name, player_stats, track, grade, category)

        # 車番（NaN対策）
        pos1_car = row.get("pos1_car_no", 5)
        pos2_car = row.get("pos2_car_no", 5)
        pos3_car = row.get("pos3_car_no", 5)

        if pd.isna(pos1_car):
            pos1_car = 5
        if pd.isna(pos2_car):
            pos2_car = 5
        if pd.isna(pos3_car):
            pos3_car = 5

        pos1_car = int(pos1_car)
        pos2_car = int(pos2_car)
        pos3_car = int(pos3_car)

        # 特徴量を構築
        features = {
            # 選手統計（1着） - 8特徴量
            "pos1_win_rate": pos1_stats["win_rate"],
            "pos1_top3_rate": pos1_stats["top3_rate"],
            "pos1_avg_payout": pos1_stats["avg_payout"],
            "pos1_high_payout_rate": pos1_stats["high_payout_rate"],
            "pos1_recent_win_rate": pos1_stats["recent_win_rate"],
            "pos1_track_win_rate": pos1_stats["track_win_rate"],
            "pos1_grade_win_rate": pos1_stats["grade_win_rate"],
            "pos1_consistency": pos1_stats["consistency"],

            # 選手統計（2着） - 8特徴量
            "pos2_win_rate": pos2_stats["win_rate"],
            "pos2_top3_rate": pos2_stats["top3_rate"],
            "pos2_avg_payout": pos2_stats["avg_payout"],
            "pos2_high_payout_rate": pos2_stats["high_payout_rate"],
            "pos2_recent_win_rate": pos2_stats["recent_win_rate"],
            "pos2_track_win_rate": pos2_stats["track_win_rate"],
            "pos2_grade_win_rate": pos2_stats["grade_win_rate"],
            "pos2_consistency": pos2_stats["consistency"],

            # 選手統計（3着） - 8特徴量
            "pos3_win_rate": pos3_stats["win_rate"],
            "pos3_top3_rate": pos3_stats["top3_rate"],
            "pos3_avg_payout": pos3_stats["avg_payout"],
            "pos3_high_payout_rate": pos3_stats["high_payout_rate"],
            "pos3_recent_win_rate": pos3_stats["recent_win_rate"],
            "pos3_track_win_rate": pos3_stats["track_win_rate"],
            "pos3_grade_win_rate": pos3_stats["grade_win_rate"],
            "pos3_consistency": pos3_stats["consistency"],

            # 3選手の統計的特徴 - 12特徴量
            "avg_win_rate": np.mean([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "std_win_rate": np.std([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "min_win_rate": np.min([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "max_win_rate": np.max([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),

            "avg_recent_win_rate": np.mean([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]]),
            "std_recent_win_rate": np.std([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]]),

            "avg_track_win_rate": np.mean([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "std_track_win_rate": np.std([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),

            "avg_high_payout_rate": np.mean([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]]),
            "std_high_payout_rate": np.std([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]]),

            "avg_consistency": np.mean([pos1_stats["consistency"], pos2_stats["consistency"], pos3_stats["consistency"]]),
            "win_rate_gap": pos1_stats["win_rate"] - pos3_stats["win_rate"],  # 力の差

            # 車番特徴 - 9特徴量
            "pos1_car_no": pos1_car,
            "pos2_car_no": pos2_car,
            "pos3_car_no": pos3_car,
            "car_sum": pos1_car + pos2_car + pos3_car,
            "car_std": np.std([pos1_car, pos2_car, pos3_car]),
            "car_range": max(pos1_car, pos2_car, pos3_car) - min(pos1_car, pos2_car, pos3_car),
            "outer_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c >= 7),
            "inner_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c <= 3),
            "has_1_car": 1 if 1 in [pos1_car, pos2_car, pos3_car] else 0,

            # グレード（簡易エンコード） - 5特徴量
            "is_F1": 1 if grade == "F1" else 0,
            "is_F2": 1 if grade == "F2" else 0,
            "is_G1": 1 if grade == "G1" else 0,
            "is_G2": 1 if grade == "G2" else 0,
            "is_G3": 1 if grade == "G3" else 0,
        }

        X_list.append(features)
        y_list.append(y)

    print(f"  完了: {len(X_list)} レース")

    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    print(f"\n  特徴量数: {X.shape[1]}")
    print(f"  高配当レース: {y.sum():,} / {len(y):,} ({y.mean()*100:.1f}%)")

    return X, y


def train_model_with_cv(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """クロスバリデーションでモデルを訓練"""
    print("\n[3/5] LightGBMモデルを学習中（5-Fold CV）...\n")

    # 最適化されたハイパーパラメータ
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'max_depth': 8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_split_gain': 0.01,
        'verbose': -1,
        'force_col_wise': True,
        'scale_pos_weight': 2.0,  # クラス不均衡対策
    }

    # 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}/5を学習中...")

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)  # ログを抑制
            ]
        )

        # 検証データでの予測
        y_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val_fold, y_pred)
        cv_scores.append(auc)
        models.append(model)

        print(f"  Fold {fold} AUC: {auc:.4f}")

    print(f"\n平均AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 最良のモデルを選択
    best_model = models[np.argmax(cv_scores)]

    return best_model, cv_scores


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """モデルを評価"""
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  精度: {report['accuracy']:.4f}")

    # 特徴量重要度Top 15
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = X_test.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\n  特徴量重要度Top 15:")
    for i, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:8.0f}")

    return {
        "auc": auc,
        "average_precision": ap,
        "classification_report": report,
        "feature_importance": importance_df.to_dict('records')
    }


def main():
    print("=" * 60)
    print("改善版予測モデルの学習（目標精度: 80%以上）")
    print("=" * 60)

    # パス設定
    csv_path = Path(__file__).parent.parent / "data" / "keirin_results_20240101_20251004.csv"
    player_stats_path = Path(__file__).parent / "models" / "player_stats_advanced.json"
    model_dir = Path(__file__).parent / "models"

    # [1/5] データ読み込み
    print("\n[1/5] データを読み込み中...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  レース数: {len(df):,}")

    with open(player_stats_path, "r", encoding="utf-8") as f:
        player_stats = json.load(f)
    print(f"  選手数: {len(player_stats):,}")

    # [2/5] 特徴量構築
    print("\n[2/5] 特徴量を構築中...")
    X, y = build_features(df, player_stats)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # [3/5] モデル学習
    model, cv_scores = train_model_with_cv(X_train, y_train)

    # [4/5] モデル評価
    print("\n[4/5] テストデータで評価中...")
    metrics = evaluate_model(model, X_test, y_test)

    # [5/5] モデル保存
    print("\n[5/5] モデルを保存中...")
    model_dir.mkdir(parents=True, exist_ok=True)

    # LightGBM形式で保存
    model.save_model(str(model_dir / "model_improved.txt"))

    # Pickle形式で保存
    with open(model_dir / "model_improved.pkl", "wb") as f:
        pickle.dump(model, f)

    # メトリクスを保存
    with open(model_dir / "metrics_improved.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # モデル情報を保存
    model_info = {
        "feature_count": X.shape[1],
        "feature_names": X.columns.tolist(),
        "cv_scores": cv_scores,
        "test_auc": metrics["auc"],
        "test_accuracy": metrics["classification_report"]["accuracy"],
    }

    with open(model_dir / "model_improved_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("\n✅ 完了！")
    print(f"  - model_improved.txt")
    print(f"  - model_improved.pkl")
    print(f"  - metrics_improved.json")
    print(f"  - model_improved_info.json")

    print("\n" + "=" * 60)
    print("🎉 改善版モデルが完成しました！")
    print(f"   テスト精度: {metrics['classification_report']['accuracy']*100:.2f}%")
    print(f"   CV平均AUC: {np.mean(cv_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
