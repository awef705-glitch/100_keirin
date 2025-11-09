#!/usr/bin/env python3
"""
競輪予測APIサーバー（高精度版）
LightGBM + シンプルなUI対応
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# グローバル変数でモデルとパラメータを保持
model = None
model_stats = None
reference_data = None


def load_model():
    """モデルとパラメータをロード"""
    global model, model_stats, reference_data

    model_dir = Path(__file__).parent / "models"

    try:
        # LightGBMモデルのロード
        model = lgb.Booster(model_file=str(model_dir / "model_lgb.txt"))
        print("✅ LightGBMモデルをロードしました")
    except Exception as e:
        print(f"⚠️  LightGBMモデルのロード失敗: {e}")
        # フォールバック: Pickleからロード
        with open(model_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Pickleモデルをロードしました")

    # 統計情報のロード
    with open(model_dir / "model_stats.json", "r", encoding="utf-8") as f:
        model_stats = json.load(f)

    # リファレンスデータのロード
    with open(model_dir / "reference_data.json", "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    print("✅ モデルとパラメータをロードしました")


def create_advanced_features(data: dict) -> dict:
    """高度な特徴量を生成"""

    car_nos = [
        float(data.get("pos1_car_no", 1)),
        float(data.get("pos2_car_no", 2)),
        float(data.get("pos3_car_no", 3))
    ]

    features = {}

    # 基本統計量
    features["pos1_car_no"] = car_nos[0]
    features["pos2_car_no"] = car_nos[1]
    features["pos3_car_no"] = car_nos[2]
    features["car_sum"] = sum(car_nos)
    features["car_std"] = float(np.std(car_nos))
    features["car_range"] = max(car_nos) - min(car_nos)
    features["car_median"] = float(np.median(car_nos))
    features["car_min"] = min(car_nos)
    features["car_max"] = max(car_nos)
    features["car_mean"] = float(np.mean(car_nos))

    # 高度な特徴量
    # 1. 連続性
    features["is_consecutive"] = int(
        abs(car_nos[1] - car_nos[0]) == 1 and
        abs(car_nos[2] - car_nos[1]) == 1
    )

    # 2. 偶奇パターン
    odd_count = sum(int(c) % 2 for c in car_nos)
    features["odd_count"] = odd_count
    features["even_count"] = 3 - odd_count
    features["all_odd"] = int(odd_count == 3)
    features["all_even"] = int(odd_count == 0)

    # 3. 分散度
    features["car_variance"] = float(np.var(car_nos))

    # 4. 大穴指標
    features["outer_count"] = sum(1 for c in car_nos if c >= 7)
    features["inner_count"] = sum(1 for c in car_nos if c <= 3)

    # 5. 車番の積
    features["car_product"] = car_nos[0] * car_nos[1] * car_nos[2]

    # 6. 車番の差
    features["diff_12"] = abs(car_nos[0] - car_nos[1])
    features["diff_23"] = abs(car_nos[1] - car_nos[2])
    features["diff_13"] = abs(car_nos[0] - car_nos[2])
    features["total_diff"] = features["diff_12"] + features["diff_23"] + features["diff_13"]

    # 7. 昇順・降順
    features["is_ascending"] = int(car_nos[0] < car_nos[1] < car_nos[2])
    features["is_descending"] = int(car_nos[0] > car_nos[1] > car_nos[2])

    # 8. レース番号
    race_no = data.get("race_no", "1")
    try:
        features["race_no_numeric"] = float(str(race_no).upper().replace("R", ""))
    except:
        features["race_no_numeric"] = 1.0

    return features


def encode_categorical(data: dict) -> dict:
    """カテゴリカル変数をエンコード"""
    encoded = {}

    for col in model_stats["cat_cols"]:
        value = data.get(col, "")
        classes = model_stats["label_encoders"].get(col, [])

        # 存在する値ならエンコード、なければ0
        if value in classes:
            encoded[f"{col}_encoded"] = float(classes.index(value))
        else:
            # デフォルト値（最頻値を0とする）
            encoded[f"{col}_encoded"] = 0.0

    return encoded


def preprocess_input(data: dict) -> pd.DataFrame:
    """入力データを前処理して特徴量を作成"""

    # デフォルト値の設定
    defaults = {
        "grade": "F1",
        "track": "京王閣",
        "category": "A級予選",
        "race_no": "1"
    }

    # デフォルト値で補完
    for key, default_value in defaults.items():
        if key not in data or not data[key]:
            data[key] = default_value

    # 高度な特徴量を生成
    numeric_features = create_advanced_features(data)

    # カテゴリカル変数をエンコード
    categorical_features = encode_categorical(data)

    # 全特徴量を結合
    all_features = {**numeric_features, **categorical_features}

    # DataFrameに変換（モデルが期待する順序で）
    feature_order = model_stats["numeric_cols"] + model_stats["cat_encoded_cols"]
    feature_values = [all_features.get(col, 0.0) for col in feature_order]

    X = pd.DataFrame([feature_values], columns=feature_order)

    return X


def generate_betting_strategy(probability: float, input_data: dict) -> dict:
    """買い方の提案を生成（改善版）"""
    car_nos = [
        int(input_data.get("pos1_car_no", 1)),
        int(input_data.get("pos2_car_no", 2)),
        int(input_data.get("pos3_car_no", 3)),
    ]

    strategy = {
        "confidence": "高" if probability > 0.65 else "中" if probability > 0.45 else "低",
        "recommendations": []
    }

    if probability > 0.65:
        # 超高確率で荒れる
        strategy["recommendations"].append({
            "type": "3連単",
            "description": f"高配当の可能性が極めて高いです（{probability*100:.1f}%）。3連単フォーメーションで勝負しましょう。",
            "suggested_numbers": car_nos,
            "bet_type": "フォーメーション",
            "priority": "★★★"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "保険として3連複も購入することをおすすめします。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス",
            "priority": "★★"
        })
    elif probability > 0.45:
        # 中程度の確率で荒れる
        strategy["recommendations"].append({
            "type": "3連複",
            "description": f"中程度の配当が期待できます（{probability*100:.1f}%）。3連複ボックスがおすすめです。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス",
            "priority": "★★★"
        })
        strategy["recommendations"].append({
            "type": "2連複",
            "description": "的中率を重視する場合は2連複も検討してください。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "フォーメーション",
            "priority": "★★"
        })
        strategy["recommendations"].append({
            "type": "3連単",
            "description": "一発を狙うなら3連単でも。",
            "suggested_numbers": car_nos,
            "bet_type": "流し",
            "priority": "★"
        })
    else:
        # 堅いレース
        strategy["recommendations"].append({
            "type": "2連複",
            "description": f"堅いレースと予想されます（荒れる確率{probability*100:.1f}%）。2連複で手堅く。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "通常",
            "priority": "★★★"
        })
        strategy["recommendations"].append({
            "type": "ワイド",
            "description": "最も確実性を求めるならワイド。",
            "suggested_numbers": car_nos[:2],
            "bet_type": "通常",
            "priority": "★★"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "少し配当を狙うなら3連複。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス",
            "priority": "★"
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
    """レース情報から予測を実行"""
    try:
        data = request.json

        # 入力の前処理
        X = preprocess_input(data)

        # 予測
        probability = float(model.predict(X)[0])
        prediction = int(probability >= 0.5)

        # 買い方の提案
        betting_strategy = generate_betting_strategy(probability, data)

        # 確信度の判定
        confidence_score = "極めて高い" if probability > 0.75 else \
                          "高い" if probability > 0.60 else \
                          "中程度" if probability > 0.40 else "低い"

        result = {
            "success": True,
            "probability": probability,
            "prediction": prediction,
            "prediction_label": "荒れる" if prediction == 1 else "堅い",
            "confidence_score": confidence_score,
            "betting_strategy": betting_strategy,
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": "LightGBM"
    })


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
