#!/usr/bin/env python3
"""
競輪高配当予測APIサーバー（全選手対応版）

iPhone単体で完結する高精度予測システム
- 全出走選手の情報を入力
- 全組み合わせを自動評価
- 高配当が期待できる買い目を推奨
"""
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from comprehensive_predictor import KerinHighPayoutPredictor

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# グローバル変数
predictor = None


def load_predictor():
    """予測システムをロード"""
    global predictor
    model_dir = Path(__file__).parent / "models"
    predictor = KerinHighPayoutPredictor(str(model_dir))


@app.route('/')
def index():
    """フロントエンドを配信"""
    return send_from_directory(app.static_folder, 'index_comprehensive.html')


@app.route('/api/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        "status": "ok",
        "predictor_loaded": predictor is not None,
        "system_type": "comprehensive (all riders)",
        "model_accuracy": "77.12%",
    })


@app.route('/api/predict-race', methods=['POST'])
def predict_race():
    """
    レース全体を分析して高配当買い目を推奨

    リクエスト形式:
    {
        "track": "平塚",
        "grade": "F1",
        "category": "一般",
        "race_no": "1R",
        "riders": [
            {"car_no": 1, "name": "山田太郎"},
            {"car_no": 2, "name": "佐藤次郎"},
            ...
        ]
    }
    """
    try:
        data = request.json

        # 必須フィールドのチェック
        required_fields = ["track", "grade", "category", "riders"]
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"必須フィールドが不足しています: {', '.join(missing_fields)}"
            }), 400

        # 選手数のチェック
        riders = data["riders"]
        if len(riders) < 7 or len(riders) > 9:
            return jsonify({
                "success": False,
                "error": f"選手数は7〜9名である必要があります（現在: {len(riders)}名）"
            }), 400

        # 選手情報のチェック
        for i, rider in enumerate(riders):
            if "car_no" not in rider or "name" not in rider:
                return jsonify({
                    "success": False,
                    "error": f"選手{i+1}の情報が不完全です（車番と名前が必要）"
                }), 400

        # 予測実行
        result = predictor.predict_race(data)

        # レスポンスを整形
        response = {
            "success": True,
            "race_info": {
                "track": data["track"],
                "grade": data["grade"],
                "category": data["category"],
                "race_no": data.get("race_no", ""),
                "num_riders": len(riders),
            },
            "analysis": {
                "chaos_level": result["race_chaos_level"],
                "avg_high_payout_probability": result["avg_high_payout_probability"],
                "total_combinations": result["total_combinations_evaluated"],
            },
            "recommendations": []
        }

        # 上位10件の推奨買い目を整形
        for i, rec in enumerate(result["top_recommendations"], 1):
            response["recommendations"].append({
                "rank": i,
                "combination": rec["combination"],
                "riders": rec["riders"],
                "car_numbers": rec["cars"],
                "high_payout_probability": rec["high_payout_probability"],
                "expected_value_score": rec["expected_value_score"],
                "rider_win_rates": [
                    {"name": rec["riders"][0], "win_rate": rec["win_rates"][0]},
                    {"name": rec["riders"][1], "win_rate": rec["win_rates"][1]},
                    {"name": rec["riders"][2], "win_rate": rec["win_rates"][2]},
                ],
                "recommendation_reason": generate_recommendation_reason(rec)
            })

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


def generate_recommendation_reason(rec: dict) -> str:
    """推奨理由を生成"""
    prob = rec["high_payout_probability"]
    win_rates = rec["win_rates"]

    reasons = []

    # 高配当確率が高い
    if prob > 0.4:
        reasons.append("高配当確率が非常に高い")
    elif prob > 0.3:
        reasons.append("高配当確率が高い")

    # 実力と人気のバランス
    avg_win_rate = sum(win_rates) / 3
    if avg_win_rate > 0.7 and prob > 0.35:
        reasons.append("実力者揃いで波乱の可能性")
    elif avg_win_rate < 0.3 and prob > 0.25:
        reasons.append("人気薄だが好走の期待")

    # 実力差
    if max(win_rates) - min(win_rates) < 0.15:
        reasons.append("実力が拮抗")
    elif max(win_rates) - min(win_rates) > 0.4:
        reasons.append("実力差が明確")

    if not reasons:
        reasons.append("期待値が高い組み合わせ")

    return "、".join(reasons)


@app.route('/api/player-search', methods=['GET'])
def player_search():
    """選手名の検索（オートコンプリート用）"""
    try:
        query = request.args.get('q', '').strip()

        if not query or len(query) < 1:
            return jsonify({
                "success": True,
                "players": []
            })

        # 選手名を検索
        matching_players = []
        for player_name, stats in predictor.player_stats.items():
            if query in player_name:
                matching_players.append({
                    "name": player_name,
                    "win_rate": stats["win_rate"],
                    "races": stats["races"],
                })

        # 勝率順でソート
        matching_players.sort(key=lambda x: x["win_rate"], reverse=True)

        # 上位20件まで
        matching_players = matching_players[:20]

        return jsonify({
            "success": True,
            "players": matching_players,
            "count": len(matching_players)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    load_predictor()
    app.run(host='0.0.0.0', port=5000, debug=True)
