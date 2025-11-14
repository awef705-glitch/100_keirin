#!/usr/bin/env python3
"""
競輪予測APIサーバー（Ultra高精度モデル対応: 98特徴量）
地域・季節（月）・曜日情報を追加（事前にわかる情報のみ使用）
"""
import json
import pickle
from pathlib import Path
from datetime import datetime

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
player_region_map = None
rider_names = None


def load_model():
    """モデルとパラメータをロード"""
    global model, model_info, player_stats, combo_stats, reference_data, player_region_map

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

    # 選手地域マッピングのロード
    with open(model_dir / "player_region_map.json", "r", encoding="utf-8") as f:
        player_region_map = json.load(f)

    # 選手名リストのロード（オートコンプリート用）
    global rider_names
    with open(model_dir / "rider_names.json", "r", encoding="utf-8") as f:
        rider_names = json.load(f)

    print("=" * 70)
    print("新モデル（Ultra）をロードしました - 脚質特徴対応（113特徴量）")
    print(f"  - 特徴量数: {model_info['feature_count']}")
    print(f"  - テストAUC: {model_info['test_auc']:.4f}")
    print(f"  - テスト精度: {model_info['test_accuracy']*100:.2f}%")
    print(f"  - 最適閾値: {model_info['optimal_threshold']}")
    print(f"  - 選手数: {len(player_stats):,}")
    print(f"  - 地域マッピング: {len(player_region_map):,}人")
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
            "nige_rate": 0.33,
            "sashi_rate": 0.33,
            "makuri_rate": 0.34,
        }

    stats = player_stats[player_name]

    # 脚質（決まり手）データを取得
    decision_dist = stats.get("decision_distribution", {})
    nige_rate = decision_dist.get("逃げ", 0.0)
    sashi_rate = decision_dist.get("差し", 0.0)
    makuri_rate = decision_dist.get("捲り", 0.0)

    features = {
        "win_rate": stats["win_rate"],
        "place_2_rate": stats["place_2_rate"],
        "place_3_rate": stats["place_3_rate"],
        "top3_rate": stats["top3_rate"],
        "avg_payout": stats["avg_payout"],
        "high_payout_rate": stats["high_payout_rate"],
        "races": min(stats["races"], 500) / 500,
        "nige_rate": nige_rate,
        "sashi_rate": sashi_rate,
        "makuri_rate": makuri_rate,
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
    """入力データを前処理して98特徴量を作成"""

    # レース情報
    track = data.get("track", "不明")
    grade = data.get("grade", "不明")
    category = data.get("category", "不明")
    meeting_day = int(data.get("meeting_day", 3))  # 何日目（1,3,5,8）

    # 日付情報（月・曜日）
    race_date = str(data.get("race_date", "20240101"))
    try:
        date_obj = datetime.strptime(race_date, "%Y%m%d")
        month = date_obj.month
        weekday = date_obj.weekday()  # 0=月曜, 6=日曜
    except:
        month = 1
        weekday = 0

    # 選手名
    pos1_name = data.get("pos1_name", "")
    pos2_name = data.get("pos2_name", "")
    pos3_name = data.get("pos3_name", "")

    # 選手の地域
    pos1_region = str(data.get("pos1_region", "不明")).strip()
    pos2_region = str(data.get("pos2_region", "不明")).strip()
    pos3_region = str(data.get("pos3_region", "不明")).strip()

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

    # 特徴量を構築（73特徴量）
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
        "meeting_day": meeting_day,

        # 脚質特徴（決まり手） - 15特徴量
        "pos1_nige_rate": pos1_stats["nige_rate"],
        "pos1_sashi_rate": pos1_stats["sashi_rate"],
        "pos1_makuri_rate": pos1_stats["makuri_rate"],
        "pos2_nige_rate": pos2_stats["nige_rate"],
        "pos2_sashi_rate": pos2_stats["sashi_rate"],
        "pos2_makuri_rate": pos2_stats["makuri_rate"],
        "pos3_nige_rate": pos3_stats["nige_rate"],
        "pos3_sashi_rate": pos3_stats["sashi_rate"],
        "pos3_makuri_rate": pos3_stats["makuri_rate"],
        "avg_nige_rate": np.mean([pos1_stats["nige_rate"], pos2_stats["nige_rate"], pos3_stats["nige_rate"]]),
        "avg_sashi_rate": np.mean([pos1_stats["sashi_rate"], pos2_stats["sashi_rate"], pos3_stats["sashi_rate"]]),
        "avg_makuri_rate": np.mean([pos1_stats["makuri_rate"], pos2_stats["makuri_rate"], pos3_stats["makuri_rate"]]),
        "nige_type_count": sum([1 if s["nige_rate"] > 0.5 else 0 for s in [pos1_stats, pos2_stats, pos3_stats]]),
        "sashi_type_count": sum([1 if s["sashi_rate"] > 0.5 else 0 for s in [pos1_stats, pos2_stats, pos3_stats]]),
        "makuri_type_count": sum([1 if s["makuri_rate"] > 0.5 else 0 for s in [pos1_stats, pos2_stats, pos3_stats]]),

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

        # 地域特徴 - 6特徴量
        "same_region_count": sum([pos1_region == pos2_region, pos2_region == pos3_region, pos1_region == pos3_region]),
        "all_same_region": 1 if (pos1_region == pos2_region == pos3_region and pos1_region != "不明") else 0,
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


@app.route('/api/rider-names', methods=['GET'])
def get_rider_names():
    """選手名リストを取得（オートコンプリート用）"""
    return jsonify(rider_names)


@app.route('/api/predict', methods=['POST'])
def predict():
    """レース全体の荒れ度を予測＋買い方提案（超高速版）"""
    try:
        data = request.json

        # レース基本情報
        track = data.get("track", "不明")
        grade = data.get("grade", "不明")
        category = data.get("category", "不明")
        meeting_day = int(data.get("meeting_day", 3))
        race_date = str(data.get("race_date", "20240101"))

        # 日付情報
        try:
            date_obj = datetime.strptime(race_date, "%Y%m%d")
            month = date_obj.month
            weekday = date_obj.weekday()
        except:
            month = 1
            weekday = 0

        # 全選手情報
        riders = data.get("riders", [])
        if len(riders) < 7:
            return jsonify({
                "success": False,
                "error": "最低7人以上の選手情報が必要です"
            }), 400

        # 各選手の統計情報を取得
        riders_stats = []
        for rider in riders:
            rider_name = rider["name"].replace(" ", "　")
            region = player_region_map.get(rider_name, "不明")
            stats = get_player_features(rider_name, track, grade, category)
            riders_stats.append({
                "car_no": rider["car_no"],
                "name": rider_name,
                "region": region,
                "stats": stats
            })

        # === レース全体の荒れ度を1回だけ予測 ===
        # 代表的な組み合わせ（上位3人）で予測
        top3_riders = sorted(riders_stats, key=lambda x: x["stats"]["win_rate"], reverse=True)[:3]

        # 代表組み合わせのデータを作成
        representative_data = {
            "track": track,
            "grade": grade,
            "category": category,
            "race_no": data.get("race_no", "1"),
            "meeting_day": meeting_day,
            "race_date": race_date,
            "pos1_car_no": top3_riders[0]["car_no"],
            "pos1_name": top3_riders[0]["name"],
            "pos1_region": top3_riders[0]["region"],
            "pos2_car_no": top3_riders[1]["car_no"],
            "pos2_name": top3_riders[1]["name"],
            "pos2_region": top3_riders[1]["region"],
            "pos3_car_no": top3_riders[2]["car_no"],
            "pos3_name": top3_riders[2]["name"],
            "pos3_region": top3_riders[2]["region"],
        }

        # 1回だけ予測実行
        X = preprocess_input(representative_data)
        base_probability = float(model.predict(X)[0])

        # レース全体の特性で補正
        avg_win_rate = np.mean([r["stats"]["win_rate"] for r in riders_stats])
        std_win_rate = np.std([r["stats"]["win_rate"] for r in riders_stats])

        # 勝率のバラつきが大きい→荒れやすい
        roughness_adjustment = std_win_rate * 0.5
        race_roughness_probability = min(base_probability + roughness_adjustment, 0.99)

        # === 脚質・地域パターンを分析 ===
        pattern_analysis = analyze_race_patterns(riders_stats, track)

        # === 買い方を提案 ===
        betting_suggestions = suggest_betting_strategy(
            race_roughness_probability,
            pattern_analysis,
            riders_stats
        )

        result = {
            "success": True,
            "race_roughness_probability": race_roughness_probability,
            "roughness_level": get_roughness_level(race_roughness_probability),
            "pattern_analysis": pattern_analysis,
            "betting_suggestions": betting_suggestions,
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


def analyze_race_patterns(riders_stats, track):
    """レースパターンを分析（脚質・地域）"""
    # 脚質分析
    nige_count = sum(1 for r in riders_stats if r["stats"]["nige_rate"] > 0.4)
    sashi_count = sum(1 for r in riders_stats if r["stats"]["sashi_rate"] > 0.4)
    makuri_count = sum(1 for r in riders_stats if r["stats"]["makuri_rate"] > 0.4)

    # 地域分析（ライン）
    regions = [r["region"] for r in riders_stats]
    from collections import Counter
    region_counts = Counter(regions)
    major_regions = [r for r, c in region_counts.items() if c >= 2]  # 2人以上の地域

    # ホーム選手
    home_riders = [r for r in riders_stats if track in r["region"] or r["region"] in track]

    # 勝率分析
    win_rates = [r["stats"]["win_rate"] for r in riders_stats]
    avg_win_rate = np.mean(win_rates)
    top_win_rate = max(win_rates)

    return {
        "nige_dominant": nige_count >= 3,
        "sashi_dominant": sashi_count >= 3,
        "makuri_dominant": makuri_count >= 3,
        "has_strong_line": len(major_regions) > 0,
        "major_regions": major_regions,
        "has_home_advantage": len(home_riders) > 0,
        "home_riders": [r["name"] for r in home_riders],
        "avg_win_rate": avg_win_rate,
        "top_win_rate": top_win_rate,
        "strength_gap": top_win_rate - avg_win_rate
    }


def suggest_betting_strategy(probability, patterns, riders_stats):
    """買い方を提案"""
    suggestions = []

    # 荒れる度合いで券種を決定
    if probability >= 0.7:
        # 超高配当が期待できる
        suggestions.append({
            "ticket_type": "3連単",
            "reason": f"荒れる確率{probability*100:.1f}% - 超高配当が期待できます",
            "strategy": "ボックス or フォーメーション",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "upset")
        })
        suggestions.append({
            "ticket_type": "3連複",
            "reason": "リスク分散しつつ高配当を狙う",
            "strategy": "ボックス",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "upset")
        })
    elif probability >= 0.5:
        # 中程度に荒れる
        suggestions.append({
            "ticket_type": "3連複",
            "reason": f"荒れる確率{probability*100:.1f}% - 適度な配当が期待",
            "strategy": "ボックス or フォーメーション",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "moderate")
        })
        suggestions.append({
            "ticket_type": "2連複",
            "reason": "的中率重視で中配当を狙う",
            "strategy": "フォーメーション",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "moderate")
        })
    else:
        # 堅い
        suggestions.append({
            "ticket_type": "2連複",
            "reason": f"荒れる確率{probability*100:.1f}% - 本命決着の可能性",
            "strategy": "軸1頭 or 軸2頭流し",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "favorite")
        })
        suggestions.append({
            "ticket_type": "ワイド",
            "reason": "確実性重視",
            "strategy": "本命サイド",
            "recommended_combinations": get_recommended_combinations(riders_stats, patterns, "favorite")
        })

    return suggestions


def get_recommended_combinations(riders_stats, patterns, strategy_type):
    """推奨する組み合わせを抽出"""
    combinations = []

    if strategy_type == "upset":
        # 荒れる場合：中穴〜大穴狙い
        # 地域ラインを重視
        if patterns["has_strong_line"]:
            for region in patterns["major_regions"]:
                line_riders = [r for r in riders_stats if r["region"] == region]
                if len(line_riders) >= 2:
                    combinations.append({
                        "cars": [r["car_no"] for r in line_riders[:3]],
                        "riders": [r["name"] for r in line_riders[:3]],
                        "reason": f"{region}ライン"
                    })

        # 差し型選手
        sashi_riders = sorted([r for r in riders_stats if r["stats"]["sashi_rate"] > 0.3],
                             key=lambda x: x["stats"]["sashi_rate"], reverse=True)[:3]
        if len(sashi_riders) >= 2:
            combinations.append({
                "cars": [r["car_no"] for r in sashi_riders],
                "riders": [r["name"] for r in sashi_riders],
                "reason": "差し型選手の組み合わせ"
            })

    elif strategy_type == "moderate":
        # 中程度：本命+穴の組み合わせ
        top_riders = sorted(riders_stats, key=lambda x: x["stats"]["win_rate"], reverse=True)[:2]
        mid_riders = sorted(riders_stats, key=lambda x: x["stats"]["win_rate"], reverse=True)[2:4]
        combinations.append({
            "cars": [top_riders[0]["car_no"], top_riders[1]["car_no"], mid_riders[0]["car_no"]],
            "riders": [top_riders[0]["name"], top_riders[1]["name"], mid_riders[0]["name"]],
            "reason": "本命2頭 + 対抗"
        })

    else:  # favorite
        # 堅い：本命中心
        top_riders = sorted(riders_stats, key=lambda x: x["stats"]["win_rate"], reverse=True)[:3]
        combinations.append({
            "cars": [r["car_no"] for r in top_riders],
            "riders": [r["name"] for r in top_riders],
            "reason": "本命上位"
        })

    return combinations[:3]  # 最大3つまで


def get_roughness_level(probability):
    """荒れ度合いのラベル"""
    if probability >= 0.7:
        return "超高配当が期待"
    elif probability >= 0.5:
        return "中〜高配当が期待"
    elif probability >= 0.3:
        return "やや荒れる可能性"
    else:
        return "本命決着の可能性"


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


# モデルを起動時にロード（gunicorn対応）
load_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
