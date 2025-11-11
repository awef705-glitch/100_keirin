#!/usr/bin/env python3
"""
全出走選手対応の高配当予測システム

このシステムの目的：
- レース前に全選手の情報を入力
- 全ての3連単組み合わせを評価
- 高配当が期待できる買い目を推奨
"""
import json
import pickle
from pathlib import Path
from itertools import permutations

import pandas as pd
import numpy as np
import lightgbm as lgb


class KerinHighPayoutPredictor:
    """競輪高配当予測システム"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent / "models"
        else:
            model_dir = Path(model_dir)

        # モデルとデータをロード
        with open(model_dir / "model_final.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(model_dir / "player_stats_advanced.json", "r", encoding="utf-8") as f:
            self.player_stats = json.load(f)

        with open(model_dir / "model_final_info.json", "r", encoding="utf-8") as f:
            self.model_info = json.load(f)

        with open(model_dir / "combo_stats.json", "r", encoding="utf-8") as f:
            combo_stats_raw = json.load(f)
            self.combo_stats = {}
            for k, v in combo_stats_raw.items():
                key = tuple(map(int, k.strip("()").split(", ")))
                self.combo_stats[key] = v

        self.optimal_threshold = 0.65

        print("=" * 70)
        print("高配当予測システム初期化完了")
        print("=" * 70)
        print(f"  登録選手数: {len(self.player_stats):,}名")
        print(f"  モデル精度: {self.model_info.get('test_accuracy', 0)*100:.2f}%")
        print(f"  特徴量数: {self.model_info['feature_count']}個")
        print("=" * 70)

    def get_player_features(self, player_name: str, track: str = None,
                           grade: str = None, category: str = None) -> dict:
        """選手の詳細特徴量を取得"""
        if player_name not in self.player_stats:
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

        stats = self.player_stats[player_name]

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

    def build_features_for_combination(self, rider1: dict, rider2: dict, rider3: dict,
                                      track: str, grade: str, category: str) -> pd.DataFrame:
        """3選手の組み合わせから特徴量を構築"""

        # 選手統計を取得
        pos1_stats = self.get_player_features(rider1["name"], track, grade, category)
        pos2_stats = self.get_player_features(rider2["name"], track, grade, category)
        pos3_stats = self.get_player_features(rider3["name"], track, grade, category)

        # 車番
        pos1_car = rider1["car_no"]
        pos2_car = rider2["car_no"]
        pos3_car = rider3["car_no"]

        # 車番組み合わせ統計
        cars_combo = tuple(sorted([pos1_car, pos2_car, pos3_car]))
        combo_high_payout_rate = self.combo_stats.get(cars_combo, 0.266)

        # 基本統計を計算
        avg_win_rate = np.mean([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]])
        avg_recent_win_rate = np.mean([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]])
        avg_high_payout_rate = np.mean([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]])
        avg_consistency = np.mean([pos1_stats["consistency"], pos2_stats["consistency"], pos3_stats["consistency"]])
        win_rate_gap_1_3 = pos1_stats["win_rate"] - pos3_stats["win_rate"]
        car_sum = pos1_car + pos2_car + pos3_car
        outer_count = sum(1 for c in [pos1_car, pos2_car, pos3_car] if c >= 7)

        # 58特徴量を構築
        features = {
            # 選手統計（1着）
            "pos1_win_rate": pos1_stats["win_rate"],
            "pos1_top3_rate": pos1_stats["top3_rate"],
            "pos1_avg_payout": pos1_stats["avg_payout"],
            "pos1_high_payout_rate": pos1_stats["high_payout_rate"],
            "pos1_recent_win_rate": pos1_stats["recent_win_rate"],
            "pos1_track_win_rate": pos1_stats["track_win_rate"],
            "pos1_grade_win_rate": pos1_stats["grade_win_rate"],
            "pos1_consistency": pos1_stats["consistency"],

            # 選手統計（2着）
            "pos2_win_rate": pos2_stats["win_rate"],
            "pos2_top3_rate": pos2_stats["top3_rate"],
            "pos2_avg_payout": pos2_stats["avg_payout"],
            "pos2_high_payout_rate": pos2_stats["high_payout_rate"],
            "pos2_recent_win_rate": pos2_stats["recent_win_rate"],
            "pos2_track_win_rate": pos2_stats["track_win_rate"],
            "pos2_grade_win_rate": pos2_stats["grade_win_rate"],
            "pos2_consistency": pos2_stats["consistency"],

            # 選手統計（3着）
            "pos3_win_rate": pos3_stats["win_rate"],
            "pos3_top3_rate": pos3_stats["top3_rate"],
            "pos3_avg_payout": pos3_stats["avg_payout"],
            "pos3_high_payout_rate": pos3_stats["high_payout_rate"],
            "pos3_recent_win_rate": pos3_stats["recent_win_rate"],
            "pos3_track_win_rate": pos3_stats["track_win_rate"],
            "pos3_grade_win_rate": pos3_stats["grade_win_rate"],
            "pos3_consistency": pos3_stats["consistency"],

            # 統計的特徴
            "avg_win_rate": avg_win_rate,
            "std_win_rate": np.std([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "min_win_rate": np.min([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "max_win_rate": np.max([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]),
            "avg_recent_win_rate": avg_recent_win_rate,
            "std_recent_win_rate": np.std([pos1_stats["recent_win_rate"], pos2_stats["recent_win_rate"], pos3_stats["recent_win_rate"]]),
            "avg_track_win_rate": np.mean([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "std_track_win_rate": np.std([pos1_stats["track_win_rate"], pos2_stats["track_win_rate"], pos3_stats["track_win_rate"]]),
            "avg_high_payout_rate": avg_high_payout_rate,
            "std_high_payout_rate": np.std([pos1_stats["high_payout_rate"], pos2_stats["high_payout_rate"], pos3_stats["high_payout_rate"]]),
            "avg_consistency": avg_consistency,

            # 実力差
            "win_rate_gap_1_2": pos1_stats["win_rate"] - pos2_stats["win_rate"],
            "win_rate_gap_2_3": pos2_stats["win_rate"] - pos3_stats["win_rate"],
            "win_rate_gap_1_3": win_rate_gap_1_3,

            # 車番特徴
            "pos1_car_no": pos1_car,
            "pos2_car_no": pos2_car,
            "pos3_car_no": pos3_car,
            "car_sum": car_sum,
            "car_std": np.std([pos1_car, pos2_car, pos3_car]),
            "car_range": max(pos1_car, pos2_car, pos3_car) - min(pos1_car, pos2_car, pos3_car),
            "outer_count": outer_count,
            "inner_count": sum(1 for c in [pos1_car, pos2_car, pos3_car] if c <= 3),
            "has_1_car": 1 if 1 in [pos1_car, pos2_car, pos3_car] else 0,
            "has_9_car": 1 if 9 in [pos1_car, pos2_car, pos3_car] else 0,

            # 車番組み合わせ統計
            "combo_high_payout_rate": combo_high_payout_rate,

            # グレード
            "is_F1": 1 if grade == "F1" else 0,
            "is_F2": 1 if grade == "F2" else 0,
            "is_G1": 1 if grade == "G1" else 0,
            "is_G2": 1 if grade == "G2" else 0,
            "is_G3": 1 if grade == "G3" else 0,

            # 交互作用特徴
            "win_rate_x_car_sum": avg_win_rate * car_sum,
            "high_payout_x_outer": avg_high_payout_rate * outer_count,
            "consistency_x_recent": avg_consistency * avg_recent_win_rate,
            "gap_x_combo": win_rate_gap_1_3 * combo_high_payout_rate,
        }

        X = pd.DataFrame([features])
        X = X[self.model_info["feature_names"]]

        return X

    def predict_race(self, race_info: dict) -> dict:
        """
        レース全体を分析して高配当買い目を推奨

        race_info = {
            "track": "平塚",
            "grade": "F1",
            "category": "一般",
            "riders": [
                {"car_no": 1, "name": "山田太郎"},
                {"car_no": 2, "name": "佐藤次郎"},
                ...
            ]
        }
        """
        print("\n" + "=" * 70)
        print(f"🏁 レース分析開始")
        print("=" * 70)
        print(f"  場名: {race_info['track']}")
        print(f"  グレード: {race_info['grade']}")
        print(f"  カテゴリー: {race_info['category']}")
        print(f"  出走選手数: {len(race_info['riders'])}名")
        print("=" * 70)

        riders = race_info["riders"]
        track = race_info["track"]
        grade = race_info["grade"]
        category = race_info["category"]

        # 全ての3連単組み合わせを評価
        print("\n📊 全組み合わせを評価中...")
        total_combinations = len(riders) * (len(riders) - 1) * (len(riders) - 2)
        print(f"  評価する組み合わせ数: {total_combinations}通り")

        results = []
        count = 0

        for perm in permutations(riders, 3):
            rider1, rider2, rider3 = perm

            # 特徴量を構築
            X = self.build_features_for_combination(
                rider1, rider2, rider3, track, grade, category
            )

            # 予測
            probability = float(self.model.predict(X, num_iteration=self.model.best_iteration)[0])

            # 各選手の勝率を取得（人気度の推定）
            r1_stats = self.get_player_features(rider1["name"], track, grade, category)
            r2_stats = self.get_player_features(rider2["name"], track, grade, category)
            r3_stats = self.get_player_features(rider3["name"], track, grade, category)

            # 人気度の推定（勝率の高い選手 = 人気）
            popularity_score = (r1_stats["win_rate"] * 3 +
                              r2_stats["win_rate"] * 2 +
                              r3_stats["win_rate"] * 1)

            # 期待値スコア = 高配当確率 × (1 / 人気度)
            # 人気薄で高配当確率が高い組み合わせが高スコア
            if popularity_score > 0:
                expected_value_score = probability / popularity_score
            else:
                expected_value_score = probability

            results.append({
                "combination": f"{rider1['car_no']}-{rider2['car_no']}-{rider3['car_no']}",
                "riders": [rider1["name"], rider2["name"], rider3["name"]],
                "cars": [rider1["car_no"], rider2["car_no"], rider3["car_no"]],
                "high_payout_probability": probability,
                "popularity_score": popularity_score,
                "expected_value_score": expected_value_score,
                "win_rates": [r1_stats["win_rate"], r2_stats["win_rate"], r3_stats["win_rate"]],
            })

            count += 1
            if count % 50 == 0:
                print(f"  進捗: {count}/{total_combinations}通り評価完了")

        print(f"  完了: {total_combinations}通り全て評価")

        # 期待値スコアでソート（高配当×穴狙い）
        results.sort(key=lambda x: x["expected_value_score"], reverse=True)

        # 上位10件を推奨
        top_recommendations = results[:10]

        # レース全体の荒れ度を計算
        avg_high_payout_prob = np.mean([r["high_payout_probability"] for r in results])
        race_chaos_level = "高" if avg_high_payout_prob > 0.35 else "中" if avg_high_payout_prob > 0.25 else "低"

        print("\n" + "=" * 70)
        print("✅ 分析完了")
        print("=" * 70)
        print(f"  レース全体の荒れ度: {race_chaos_level} ({avg_high_payout_prob*100:.1f}%)")
        print(f"  推奨買い目数: {len(top_recommendations)}通り")
        print("=" * 70)

        return {
            "race_chaos_level": race_chaos_level,
            "avg_high_payout_probability": avg_high_payout_prob,
            "top_recommendations": top_recommendations,
            "total_combinations_evaluated": total_combinations,
        }


def main():
    """テスト実行"""
    predictor = KerinHighPayoutPredictor()

    # テストデータ
    race_info = {
        "track": "平塚",
        "grade": "F1",
        "category": "一般",
        "riders": [
            {"car_no": 1, "name": "梅川　風子"},
            {"car_no": 2, "name": "児玉　碧衣"},
            {"car_no": 3, "name": "尾方　真生"},
            {"car_no": 4, "name": "佐藤　水菜"},
            {"car_no": 5, "name": "仲澤　春香"},
            {"car_no": 6, "name": "市田龍生都"},
            {"car_no": 7, "name": "山崎　歩夢"},
        ]
    }

    result = predictor.predict_race(race_info)

    print("\n" + "=" * 70)
    print("💰 推奨買い目 Top 10")
    print("=" * 70)

    for i, rec in enumerate(result["top_recommendations"], 1):
        print(f"\n{i}. {rec['combination']}")
        print(f"   選手: {rec['riders'][0]} → {rec['riders'][1]} → {rec['riders'][2]}")
        print(f"   高配当確率: {rec['high_payout_probability']*100:.1f}%")
        print(f"   期待値スコア: {rec['expected_value_score']:.4f}")
        print(f"   勝率: {rec['win_rates'][0]*100:.1f}% / {rec['win_rates'][1]*100:.1f}% / {rec['win_rates'][2]*100:.1f}%")


if __name__ == "__main__":
    main()
