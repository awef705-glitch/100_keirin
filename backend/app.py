#!/usr/bin/env python3
"""
競輪予測APIサーバー（Ultra高精度モデル対応）
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# グローバル変数でモデルとパラメータを保持
model = None
model_info = None
player_stats = None
combo_stats = None
reference_data = None


def load_model():
    """モデルとパラメータをロード"""
    global model, model_info, player_stats, combo_stats, reference_data

    model_dir = Path(__file__).parent / "models"

    # 新モデル（Ultra）のロード
    with open(model_dir / "model_ultra.pkl", "rb") as f:
        model = pickle.load(f)

    # モデル情報のロード
    with open(model_dir / "model_ultra_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)

    # 選手統計のロード
    with open(model_dir / "player_stats_advanced.json", "r", encoding="utf-8") as f:
        player_stats = json.load(f)

    # 車番組み合わせ統計のロード
    with open(model_dir / "combo_stats.json", "r", encoding="utf-8") as f:
        combo_stats_raw = json.load(f)
        combo_stats = {}
        for k, v in combo_stats_raw.items():
            key = tuple(map(int, k.strip("()").split(", ")))
            combo_stats[key] = v

    # リファレンスデータのロード
    with open(model_dir / "reference_data.json", "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    print("=" * 70)
    print("新モデル（Ultra）をロードしました")
    print(f"  - 特徴量数: {model_info['feature_count']}")
    print(f"  - テストAUC: {model_info['test_auc']:.4f}")
    print(f"  - テスト精度: {model_info['test_accuracy']*100:.2f}%")
    print(f"  - 最適閾値: {model_info['optimal_threshold']}")
    print(f"  - 選手数: {len(player_stats):,}")
    print("=" * 70)


def get_player_features(player_name: str, track: str = None, grade: str = None, category: str = None) -> dict:
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


def preprocess_input(data: dict) -> pd.DataFrame:
    """入力データを前処理して72特徴量を作成"""

    # レース情報
    track = data.get("track", "不明")
    grade = data.get("grade", "不明")
    category = data.get("category", "不明")

    # 選手名
    pos1_name = data.get("pos1_name", "")
    pos2_name = data.get("pos2_name", "")
    pos3_name = data.get("pos3_name", "")

    # 選手統計を取得
    pos1_stats = get_player_features(pos1_name, track, grade, category)
    pos2_stats = get_player_features(pos2_name, track, grade, category)
    pos3_stats = get_player_features(pos3_name, track, grade, category)

    # 車番
    pos1_car = int(data.get("pos1_car_no", 5))
    pos2_car = int(data.get("pos2_car_no", 5))
    pos3_car = int(data.get("pos3_car_no", 5))

    # 車番組み合わせ統計
    cars_combo = tuple(sorted([pos1_car, pos2_car, pos3_car]))
    combo_high_payout_rate = combo_stats.get(cars_combo, 0.266)

    # 基本統計を計算
    win_rates = [pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]]
    avg_win_rate = np.mean(win_rates)
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

    # 特徴量を構築（72特徴量）
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

    # DataFrameに変換して特徴量順序を保証
    X = pd.DataFrame([features])
    X = X[model_info["feature_names"]]

    return X


def generate_betting_strategy(probability: float, input_data: dict) -> dict:
    """買い方の提案を生成（Ultra高精度モデル用）"""
    car_nos = [
        int(input_data.get("pos1_car_no", 0)),
        int(input_data.get("pos2_car_no", 0)),
        int(input_data.get("pos3_car_no", 0)),
    ]

    player_names = [
        input_data.get("pos1_name", "不明"),
        input_data.get("pos2_name", "不明"),
        input_data.get("pos3_name", "不明"),
    ]

    # 選手の勝率情報を取得
    track = input_data.get("track", "不明")
    grade = input_data.get("grade", "不明")
    pos1_stats = get_player_features(player_names[0], track, grade)
    pos2_stats = get_player_features(player_names[1], track, grade)
    pos3_stats = get_player_features(player_names[2], track, grade)

    avg_win_rate = np.mean([pos1_stats["win_rate"], pos2_stats["win_rate"], pos3_stats["win_rate"]])

    strategy = {
        "confidence": "高" if probability > 0.7 else "中" if probability > 0.5 else "低",
        "recommendations": [],
        "player_info": [
            {
                "position": "1番手",
                "name": player_names[0],
                "car_no": car_nos[0],
                "win_rate": f"{pos1_stats['win_rate']*100:.1f}%",
                "high_payout_rate": f"{pos1_stats['high_payout_rate']*100:.1f}%"
            },
            {
                "position": "2番手",
                "name": player_names[1],
                "car_no": car_nos[1],
                "win_rate": f"{pos2_stats['win_rate']*100:.1f}%",
                "high_payout_rate": f"{pos2_stats['high_payout_rate']*100:.1f}%"
            },
            {
                "position": "3番手",
                "name": player_names[2],
                "car_no": car_nos[2],
                "win_rate": f"{pos3_stats['win_rate']*100:.1f}%",
                "high_payout_rate": f"{pos3_stats['high_payout_rate']*100:.1f}%"
            }
        ]
    }

    if probability > 0.65:
        # 高確率で荒れる場合
        strategy["recommendations"].append({
            "type": "3連単",
            "description": f"高配当が期待できます（確率{probability*100:.1f}%）。3連単でのボックス買いを検討してください。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス",
            "reason": f"選手の平均勝率{avg_win_rate*100:.1f}%と組み合わせから高配当の可能性が高いです。"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "リスクを抑えつつ配当を狙う場合は3連複もおすすめです。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス"
        })
    elif probability > 0.45:
        # 中程度の確率
        strategy["recommendations"].append({
            "type": "2連複",
            "description": f"中程度の配当が期待できます（確率{probability*100:.1f}%）。2連複での的中率を重視した買い方が良いでしょう。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "フォーメーション"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "やや配当を狙う場合は3連複も検討してください。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス"
        })
    else:
        # 低確率
        strategy["recommendations"].append({
            "type": "2連複",
            "description": f"堅いレースと予想されます（荒れる確率{probability*100:.1f}%）。的中率重視の2連複がおすすめです。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "通常"
        })
        strategy["recommendations"].append({
            "type": "ワイド",
            "description": "確実性を求める場合はワイドでの購入を検討してください。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "通常"
        })

    return strategy


@app.route('/')
def index():
    """フロントエンドを配信"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/reference-data', methods=['GET'])
def get_reference_data():
    """リファレンスデータを取得"""
    return jsonify(reference_data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """レース情報から予測を実行（Ultra高精度モデル）"""
    try:
        data = request.json

        # 入力の前処理（72特徴量を計算）
        X = preprocess_input(data)

        # 予測（LightGBMモデル）
        probability = float(model.predict(X)[0])

        # 最適閾値（0.65）で判定
        prediction = int(probability >= model_info["optimal_threshold"])

        # 買い方の提案
        betting_strategy = generate_betting_strategy(probability, data)

        result = {
            "success": True,
            "probability": probability,
            "prediction": prediction,
            "prediction_label": "荒れる" if prediction == 1 else "堅い",
            "betting_strategy": betting_strategy,
            "model_info": {
                "model_type": "LightGBM Ultra",
                "test_auc": model_info["test_auc"],
                "test_accuracy": model_info["test_accuracy"],
                "feature_count": model_info["feature_count"]
            }
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": "LightGBM Ultra" if model is not None else None,
        "feature_count": model_info["feature_count"] if model_info else None,
        "test_auc": model_info["test_auc"] if model_info else None,
    })


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
