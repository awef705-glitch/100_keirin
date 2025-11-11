#!/usr/bin/env python3
"""
競輪予測APIサーバー（レース前予測版）
選手の過去成績を使用した本当の予測モデル
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

# グローバル変数でモデルとデータを保持
model = None
player_stats = None
model_info = None
combo_stats = None
optimal_threshold = 0.65


def load_model():
    """モデルと選手統計をロード"""
    global model, player_stats, model_info, combo_stats, optimal_threshold

    model_dir = Path(__file__).parent / "models"

    # モデルのロード
    with open(model_dir / "model_final.pkl", "rb") as f:
        model = pickle.load(f)

    # 選手統計のロード（詳細版）
    with open(model_dir / "player_stats_advanced.json", "r", encoding="utf-8") as f:
        player_stats = json.load(f)

    # モデル情報のロード
    with open(model_dir / "model_final_info.json", "r", encoding="utf-8") as f:
        model_info = json.load(f)

    # 車番組み合わせ統計のロード
    with open(model_dir / "combo_stats.json", "r", encoding="utf-8") as f:
        combo_stats_raw = json.load(f)
        # キーをタプルに変換
        combo_stats = {}
        for k, v in combo_stats_raw.items():
            # "(1, 2, 3)" のような文字列をタプルに変換
            key = tuple(map(int, k.strip("()").split(", ")))
            combo_stats[key] = v

    # 最適閾値のロード
    try:
        with open(model_dir / "optimal_threshold.json", "r", encoding="utf-8") as f:
            threshold_info = json.load(f)
            optimal_threshold = threshold_info.get("threshold", 0.65)
    except:
        optimal_threshold = 0.65

    print(f"改善版モデルをロードしました")
    print(f"  特徴量数: {model_info['feature_count']}")
    print(f"  登録選手数: {len(player_stats):,}")
    print(f"  車番組み合わせ: {len(combo_stats):,}")
    print(f"  予測精度: {model_info.get('test_accuracy', 0)*100:.2f}%")
    print(f"  最適閾値: {optimal_threshold:.2f}")


def get_player_features(player_name: str, track: str = None, grade: str = None, category: str = None) -> dict:
    """選手の過去成績から特徴量を取得（詳細版）"""
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

    # 安定性
    features["consistency"] = 1.0 - abs(features["recent_win_rate"] - stats["win_rate"])

    return features


def preprocess_input(data: dict) -> tuple:
    """入力データを前処理して特徴量を作成（最終版）"""

    # レース情報を取得
    track = data.get("track", "不明")
    grade = data.get("grade", "不明")
    category = data.get("category", "不明")

    # 選手名を取得
    pos1_name = data.get("pos1_name", "")
    pos2_name = data.get("pos2_name", "")
    pos3_name = data.get("pos3_name", "")

    # 選手の過去成績を取得（場所・グレード考慮）
    pos1_stats = get_player_features(pos1_name, track, grade, category)
    pos2_stats = get_player_features(pos2_name, track, grade, category)
    pos3_stats = get_player_features(pos3_name, track, grade, category)

    # 車番を取得
    pos1_car = int(data.get("pos1_car_no", 5))
    pos2_car = int(data.get("pos2_car_no", 5))
    pos3_car = int(data.get("pos3_car_no", 5))

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

    # 特徴量を構築（最終版 - 58特徴量）
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

        # 3選手の統計的特徴 - 14特徴量
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

        # 実力差 - 3特徴量
        "win_rate_gap_1_2": pos1_stats["win_rate"] - pos2_stats["win_rate"],
        "win_rate_gap_2_3": pos2_stats["win_rate"] - pos3_stats["win_rate"],
        "win_rate_gap_1_3": win_rate_gap_1_3,

        # 車番特徴 - 10特徴量
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

        # 車番組み合わせ統計 - 1特徴量
        "combo_high_payout_rate": combo_high_payout_rate,

        # グレード（簡易エンコード） - 5特徴量
        "is_F1": 1 if grade == "F1" else 0,
        "is_F2": 1 if grade == "F2" else 0,
        "is_G1": 1 if grade == "G1" else 0,
        "is_G2": 1 if grade == "G2" else 0,
        "is_G3": 1 if grade == "G3" else 0,

        # 交互作用特徴 - 4特徴量
        "win_rate_x_car_sum": avg_win_rate * car_sum,
        "high_payout_x_outer": avg_high_payout_rate * outer_count,
        "consistency_x_recent": avg_consistency * avg_recent_win_rate,
        "gap_x_combo": win_rate_gap_1_3 * combo_high_payout_rate,
    }

    # DataFrameに変換し、モデルの特徴量順に並べる
    X = pd.DataFrame([features])
    X = X[model_info["feature_names"]]

    return X, {
        "pos1_stats": pos1_stats,
        "pos2_stats": pos2_stats,
        "pos3_stats": pos3_stats,
        "pos1_known": pos1_name in player_stats,
        "pos2_known": pos2_name in player_stats,
        "pos3_known": pos3_name in player_stats,
    }


def generate_betting_strategy(probability: float, input_data: dict, stats_info: dict) -> dict:
    """買い方の提案を生成"""
    car_nos = [
        int(input_data.get("pos1_car_no", 0)),
        int(input_data.get("pos2_car_no", 0)),
        int(input_data.get("pos3_car_no", 0)),
    ]

    strategy = {
        "confidence": "高" if probability > 0.4 else "中" if probability > 0.3 else "低",
        "recommendations": [],
        "player_analysis": {
            "pos1": {
                "name": input_data.get("pos1_name", ""),
                "win_rate": f"{stats_info['pos1_stats']['win_rate']*100:.1f}%",
                "known": stats_info["pos1_known"]
            },
            "pos2": {
                "name": input_data.get("pos2_name", ""),
                "win_rate": f"{stats_info['pos2_stats']['win_rate']*100:.1f}%",
                "known": stats_info["pos2_known"]
            },
            "pos3": {
                "name": input_data.get("pos3_name", ""),
                "win_rate": f"{stats_info['pos3_stats']['win_rate']*100:.1f}%",
                "known": stats_info["pos3_known"]
            },
        }
    }

    if probability > 0.35:
        # 高確率で荒れる場合
        strategy["recommendations"].append({
            "type": "3連単",
            "description": "高配当が期待できます。3連単での買い目を検討してください。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックスまたはフォーメーション"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "リスクを抑えつつ配当を狙う場合は3連複もおすすめです。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス"
        })
    elif probability > 0.25:
        # 中程度の確率
        strategy["recommendations"].append({
            "type": "2連複",
            "description": "中程度の配当が期待できます。2連複での的中率を重視した買い方が良いでしょう。",
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
            "description": "堅いレースと予想されます。的中率重視の2連複がおすすめです。",
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


@app.route('/api/players', methods=['GET'])
def get_players():
    """登録されている選手の一覧を取得"""
    # 選手名のリストを返す（勝率順）
    sorted_players = sorted(
        player_stats.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True
    )
    return jsonify({
        "players": [name for name, _ in sorted_players],
        "count": len(player_stats)
    })


@app.route('/api/player/<player_name>', methods=['GET'])
def get_player_stats(player_name):
    """特定選手の統計を取得"""
    if player_name in player_stats:
        return jsonify({
            "success": True,
            "player_name": player_name,
            "stats": player_stats[player_name]
        })
    else:
        return jsonify({
            "success": False,
            "error": "選手が見つかりません"
        }), 404


@app.route('/api/predict', methods=['POST'])
def predict():
    """レース情報から予測を実行"""
    try:
        data = request.json

        # 必須フィールドのチェック
        required_fields = ["pos1_name", "pos2_name", "pos3_name",
                          "pos1_car_no", "pos2_car_no", "pos3_car_no"]
        missing_fields = [f for f in required_fields if not data.get(f)]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"必須フィールドが不足しています: {', '.join(missing_fields)}"
            }), 400

        # 入力の前処理
        X, stats_info = preprocess_input(data)

        # 予測（最適閾値を使用）
        probability = float(model.predict(X, num_iteration=model.best_iteration)[0])
        prediction = int(probability >= optimal_threshold)

        # 買い方の提案
        betting_strategy = generate_betting_strategy(probability, data, stats_info)

        result = {
            "success": True,
            "probability": probability,
            "prediction": prediction,
            "prediction_label": "荒れる（高配当）" if prediction == 1 else "堅い（低配当）",
            "betting_strategy": betting_strategy,
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
        "player_count": len(player_stats) if player_stats else 0,
        "model_type": "improved (before-race data)",
        "model_accuracy": f"{model_info.get('test_accuracy', 0)*100:.2f}%" if model_info else "N/A",
        "optimal_threshold": optimal_threshold,
        "feature_count": model_info.get("feature_count", 0) if model_info else 0,
    })


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
