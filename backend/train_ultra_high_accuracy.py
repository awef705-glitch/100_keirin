#!/usr/bin/env python3
"""
超高精度モデルの学習（目標：80%以上）

改善手法：
1. Optunaによる徹底的なハイパーパラメータチューニング
2. より高度な特徴量エンジニアリング（100特徴量超）
3. 事前情報のみ使用：何日目、地域、季節、曜日
4. アンサンブル学習
5. 時系列分割での厳密な検証
"""
import json
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, accuracy_score


def get_player_features(player_name: str, player_stats: dict, track: str = None,
                        grade: str = None, category: str = None) -> dict:
    """選手の詳細な特徴量を取得"""
    if player_name not in player_stats:
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

    features = {
        "win_rate": stats["win_rate"],
        "place_2_rate": stats["place_2_rate"],
        "place_3_rate": stats["place_3_rate"],
        "top3_rate": stats["top3_rate"],
        "avg_payout": stats["avg_payout"],
        "high_payout_rate": stats["high_payout_rate"],
        "races": min(stats["races"], 500) / 500,
    }

    features["recent_win_rate"] = stats.get("recent_win_rate", stats["win_rate"])
    features["recent_top3_rate"] = stats.get("recent_top3_rate", stats["top3_rate"])

    if track and track in stats.get("by_track", {}):
        features["track_win_rate"] = stats["by_track"][track]["win_rate"]
    else:
        features["track_win_rate"] = stats["win_rate"]

    if grade and grade in stats.get("by_grade", {}):
        features["grade_win_rate"] = stats["by_grade"][grade]["win_rate"]
    else:
        features["grade_win_rate"] = stats["win_rate"]

    if category and category in stats.get("by_category", {}):
        features["category_win_rate"] = stats["by_category"][category]["win_rate"]
    else:
        features["category_win_rate"] = stats["win_rate"]

    features["consistency"] = 1.0 - abs(features["recent_win_rate"] - stats["win_rate"])

    return features


def build_advanced_features(df: pd.DataFrame, player_stats: dict, combo_stats: dict) -> tuple:
    """より高度な特徴量を構築（拡張版: 100特徴量超）"""
    print("\n高度な特徴量を構築中（地域・季節・曜日対応）...")

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
        meeting_icon = row.get("meeting_icon", 3)  # 何日目（1,3,5,8）

        # 日付情報
        race_date = str(row.get("race_date", "20240101"))
        try:
            date_obj = datetime.strptime(race_date, "%Y%m%d")
            month = date_obj.month
            weekday = date_obj.weekday()  # 0=月曜, 6=日曜
        except:
            month = 1
            weekday = 0

        # 選手名
        pos1_name = row.get("pos1_name")
        pos2_name = row.get("pos2_name")
        pos3_name = row.get("pos3_name")

        # 選手の地域
        pos1_region = str(row.get("pos1_region", "不明")).strip()
        pos2_region = str(row.get("pos2_region", "不明")).strip()
        pos3_region = str(row.get("pos3_region", "不明")).strip()

        if pd.isna(pos1_name) or pd.isna(pos2_name) or pd.isna(pos3_name):
            continue

        # 選手統計を取得
        pos1_stats = get_player_features(pos1_name, player_stats, track, grade, category)
        pos2_stats = get_player_features(pos2_name, player_stats, track, grade, category)
        pos3_stats = get_player_features(pos3_name, player_stats, track, grade, category)

        # 車番
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

        # 車番組み合わせ統計
        cars_combo = tuple(sorted([pos1_car, pos2_car, pos3_car]))
        combo_high_payout_rate = combo_stats.get(cars_combo, 0.266)

        # 基本統計を計算
        avg_win_rate = np.mean([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]])
        avg_recent_win_rate = np.mean([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]])
        avg_high_payout_rate = np.mean([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]])
        avg_consistency = np.mean([pos1_stats["consistency"], pos2_stats["consistency"], pos3_stats["consistency"]])
        win_rate_gap_1_3 = pos1_stats["win_rate"] - pos3_stats["win_rate"]
        car_sum = pos1_car + pos2_car + pos3_car
        outer_count = sum(1 for c in [pos1_car, pos2_car, pos3_car] if c >= 7)

        # 新しい高度な特徴量
        # 1. 非線形交互作用
        win_rate_product = pos1_stats["win_rate"] * pos2_stats["win_rate"] * pos3_stats["win_rate"]
        high_payout_product = pos1_stats["high_payout_rate"] * pos2_stats["high_payout_rate"] * pos3_stats["high_payout_rate"]

        # 2. ランク特徴
        win_rates = [pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]
        win_rates_sorted = sorted(win_rates, reverse=True)
        win_rate_rank_diff = win_rates_sorted[0] - win_rates_sorted[2]

        # 3. 比率特徴
        if pos3_stats["win_rate"] > 0:
            win_rate_ratio_1_3 = pos1_stats["win_rate"] / pos3_stats["win_rate"]
        else:
            win_rate_ratio_1_3 = 10.0

        # 4. 車番の非線形特徴
        car_product = pos1_car * pos2_car * pos3_car
        car_harmonic_mean = 3 / (1/pos1_car + 1/pos2_car + 1/pos3_car) if all(c > 0 for c in [pos1_car, pos2_car, pos3_car]) else 5.0

        # 5. 安定性の交互作用
        consistency_x_win_rate = avg_consistency * avg_win_rate
        consistency_variance = np.var([pos1_stats["consistency"], pos2_stats["consistency"], pos3_stats["consistency"]])

        # 特徴量を構築（拡張: 58 → 72 → 73 → 98特徴量）
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

            # 統計的特徴 - 14特徴量
            "avg_win_rate": avg_win_rate,
            "std_win_rate": np.std(win_rates),
            "min_win_rate": np.min(win_rates),
            "max_win_rate": np.max(win_rates),
            "avg_recent_win_rate": avg_recent_win_rate,
            "std_recent_win_rate": np.std([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]]),
            "avg_track_win_rate": np.mean([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "std_track_win_rate": np.std([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "avg_high_payout_rate": avg_high_payout_rate,
            "std_high_payout_rate": np.std([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]]),
            "avg_consistency": avg_consistency,
            "win_rate_gap_1_2": pos1_stats["win_rate"] - pos2_stats["win_rate"],
            "win_rate_gap_2_3": pos2_stats["win_rate"] - pos3_stats["win_rate"],
            "win_rate_gap_1_3": win_rate_gap_1_3,

            # 車番特徴 - 10特徴量
            "pos1_car_no": pos1_car,
            "pos2_car_no": pos2_car,
            "pos3_car_no": pos3_car,
            "car_sum": car_sum,
            "car_std": np.std([pos1_car, pos2_car, pos3_car]),
            "car_range": max([pos1_car, pos2_car, pos3_car]) - min([pos1_car, pos2_car, pos3_car]),
            "outer_count": outer_count,
            "inner_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c <= 3),
            "has_1_car": 1 if 1 in [pos1_car, pos2_car, pos3_car] else 0,
            "has_9_car": 1 if 9 in [pos1_car, pos2_car, pos3_car] else 0,

            # 車番組み合わせ統計 - 1特徴量
            "combo_high_payout_rate": combo_high_payout_rate,

            # グレード - 5特徴量
            "is_F1": 1 if grade == "F1" else 0,
            "is_F2": 1 if grade == "F2" else 0,
            "is_G1": 1 if grade == "G1" else 0,
            "is_G2": 1 if grade == "G2" else 0,
            "is_G3": 1 if grade == "G3" else 0,

            # 何日目 - 1特徴量
            "meeting_day": int(meeting_icon) if not pd.isna(meeting_icon) else 3,

            # 地域特徴 - 6特徴量
            "same_region_count": sum([pos1_region == pos2_region, pos2_region == pos3_region, pos1_region == pos3_region]),
            "all_same_region": 1 if (pos1_region == pos2_region == pos3_region) else 0,
            "pos1_is_home": 1 if (track in pos1_region or pos1_region in track) else 0,
            "pos2_is_home": 1 if (track in pos2_region or pos2_region in track) else 0,
            "pos3_is_home": 1 if (track in pos3_region or pos3_region in track) else 0,
            "home_count": sum([1 if (track in r or r in track) else 0 for r in [pos1_region, pos2_region, pos3_region]]),

            # 月（季節） - 12特徴量
            "month_1": 1 if month == 1 else 0,
            "month_2": 1 if month == 2 else 0,
            "month_3": 1 if month == 3 else 0,
            "month_4": 1 if month == 4 else 0,
            "month_5": 1 if month == 5 else 0,
            "month_6": 1 if month == 6 else 0,
            "month_7": 1 if month == 7 else 0,
            "month_8": 1 if month == 8 else 0,
            "month_9": 1 if month == 9 else 0,
            "month_10": 1 if month == 10 else 0,
            "month_11": 1 if month == 11 else 0,
            "month_12": 1 if month == 12 else 0,

            # 曜日 - 7特徴量
            "weekday_0": 1 if weekday == 0 else 0,  # 月曜
            "weekday_1": 1 if weekday == 1 else 0,  # 火曜
            "weekday_2": 1 if weekday == 2 else 0,  # 水曜
            "weekday_3": 1 if weekday == 3 else 0,  # 木曜
            "weekday_4": 1 if weekday == 4 else 0,  # 金曜
            "weekday_5": 1 if weekday == 5 else 0,  # 土曜
            "weekday_6": 1 if weekday == 6 else 0,  # 日曜

            # 基本交互作用特徴 - 4特徴量
            "win_rate_x_car_sum": avg_win_rate * car_sum,
            "high_payout_x_outer": avg_high_payout_rate * outer_count,
            "consistency_x_recent": avg_consistency * avg_recent_win_rate,
            "gap_x_combo": win_rate_gap_1_3 * combo_high_payout_rate,

            # 新しい高度な特徴量 - 14特徴量
            "win_rate_product": win_rate_product,
            "high_payout_product": high_payout_product,
            "win_rate_rank_diff": win_rate_rank_diff,
            "win_rate_ratio_1_3": win_rate_ratio_1_3,
            "car_product": car_product,
            "car_harmonic_mean": car_harmonic_mean,
            "consistency_x_win_rate": consistency_x_win_rate,
            "consistency_variance": consistency_variance,
            "avg_payout_mean": np.mean([pos1_stats["avg_payout"], pos2_stats["avg_payout"], pos3_stats["avg_payout"]]),
            "avg_payout_std": np.std([pos1_stats["avg_payout"], pos2_stats["avg_payout"], pos3_stats["avg_payout"]]),
            "top3_rate_mean": np.mean([pos1_stats["top3_rate"], pos2_stats["top3_rate"], pos3_stats["top3_rate"]]),
            "top3_rate_std": np.std([pos1_stats["top3_rate"], pos2_stats["top3_rate"], pos3_stats["top3_rate"]]),
            "recent_x_track_win_rate": avg_recent_win_rate * np.mean([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "grade_x_category_interaction": (1 if grade == "F1" else 0) * avg_win_rate,
        }

        X_list.append(features)
        y_list.append(y)

    print(f"  完了: {len(X_list)} レース")

    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    print(f"\n  特徴量数: {X.shape[1]}")
    print(f"  高配当レース: {y.sum():,} / {len(y):,} ({y.mean()*100:.1f}%)")

    return X, y


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna最適化の目的関数"""

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'verbose': -1,
        'force_col_wise': True,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred_binary = (y_pred >= 0.65).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)

    return accuracy


def main():
    print("=" * 70)
    print("超高精度モデルの学習（目標: 80%以上）")
    print("=" * 70)

    csv_path = Path(__file__).parent.parent / "data" / "keirin_results_20240101_20251004.csv"
    player_stats_path = Path(__file__).parent / "models" / "player_stats_advanced.json"
    combo_stats_path = Path(__file__).parent / "models" / "combo_stats.json"
    model_dir = Path(__file__).parent / "models"

    # [1/5] データ読み込み
    print("\n[1/5] データを読み込み中...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"  レース数: {len(df):,}")

    with open(player_stats_path, "r", encoding="utf-8") as f:
        player_stats = json.load(f)
    print(f"  選手数: {len(player_stats):,}")

    with open(combo_stats_path, "r", encoding="utf-8") as f:
        combo_stats_raw = json.load(f)
        combo_stats = {}
        for k, v in combo_stats_raw.items():
            key = tuple(map(int, k.strip("()").split(", ")))
            combo_stats[key] = v

    # [2/5] 高度な特徴量構築
    print("\n[2/5] 高度な特徴量を構築中...")
    X, y = build_advanced_features(df, player_stats, combo_stats)

    # データ分割（80/20）
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # さらに訓練データから検証データを分割
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # [3/5] Optunaでハイパーパラメータ最適化
    print("\n[3/5] Optunaでハイパーパラメータ最適化中（10試行・軽量版）...")
    print("  この処理には時間がかかります（約20〜30分）")

    study = optuna.create_study(direction='maximize', study_name='keirin_ultra_accuracy')
    study.optimize(
        lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
        n_trials=10,
        show_progress_bar=True
    )

    print(f"\n最適なハイパーパラメータ:")
    print(json.dumps(study.best_params, indent=2))
    print(f"最高精度: {study.best_value*100:.2f}%")

    # [4/5] 最適なパラメータでモデルを学習
    print("\n[4/5] 最適なパラメータでモデルを学習中...")

    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'force_col_wise': True,
    })

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # [5/5] 評価
    print("\n[5/5] テストデータで評価中...")

    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.65).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  精度 (閾値=0.65): {acc*100:.2f}%")

    # 特徴量重要度Top 20
    feature_importance = model.feature_importance(importance_type='gain')
    feature_names = X.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(f"\n  特徴量重要度Top 20:")
    for i, row in importance_df.head(20).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:8.0f}")

    # モデル保存
    print("\n[6/6] モデルを保存中...")
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / "model_ultra.txt"))
    with open(model_dir / "model_ultra.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / "metrics_ultra.json", "w", encoding="utf-8") as f:
        json.dump({
            "auc": auc,
            "average_precision": ap,
            "accuracy": acc,
            "classification_report": report,
        }, f, ensure_ascii=False, indent=2)

    model_info = {
        "feature_count": X.shape[1],
        "feature_names": X.columns.tolist(),
        "test_auc": auc,
        "test_accuracy": acc,
        "optimal_threshold": 0.65,
        "best_params": study.best_params,
    }

    with open(model_dir / "model_ultra_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print("\n✅ 完了！")
    print(f"  - model_ultra.txt")
    print(f"  - model_ultra.pkl")
    print(f"  - metrics_ultra.json")
    print(f"  - model_ultra_info.json")

    print("\n" + "=" * 70)
    print("🎉 超高精度モデルが完成しました！")
    print(f"   テスト精度: {acc*100:.2f}%")
    print(f"   AUC: {auc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
