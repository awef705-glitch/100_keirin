#!/usr/bin/env python3
"""
競輪予測APIサーバー
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
feature_names = None
standardization_params = None
categorical_columns = None
reference_data = None


def load_model():
    """モデルとパラメータをロード"""
    global model, feature_names, standardization_params, categorical_columns, reference_data

    model_dir = Path(__file__).parent / "models"

    # モデルのロード
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    # 特徴量名のロード
    with open(model_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # 標準化パラメータのロード
    with open(model_dir / "standardization_params.json", "r", encoding="utf-8") as f:
        standardization_params = json.load(f)

    # カテゴリカル列のロード
    with open(model_dir / "categorical_columns.json", "r", encoding="utf-8") as f:
        categorical_columns = json.load(f)

    # リファレンスデータのロード
    with open(model_dir / "reference_data.json", "r", encoding="utf-8") as f:
        reference_data = json.load(f)

    print("モデルとパラメータをロードしました")


def preprocess_input(data: dict) -> pd.DataFrame:
    """入力データを前処理して特徴量を作成"""
    # 車番の統計量を計算
    car_nos = [
        float(data.get("pos1_car_no", 0)),
        float(data.get("pos2_car_no", 0)),
        float(data.get("pos3_car_no", 0)),
    ]

    numeric_features = {
        "car_sum": sum(car_nos),
        "car_std": np.std(car_nos),
        "car_range": max(car_nos) - min(car_nos),
        "car_median": np.median(car_nos),
        "car_min": min(car_nos),
        "car_max": max(car_nos),
    }

    # 標準化
    for col, value in numeric_features.items():
        params = standardization_params[col]
        if params["std"] == 0:
            numeric_features[col] = 0
        else:
            numeric_features[col] = (value - params["mean"]) / params["std"]

    # カテゴリカル特徴量
    cat_data = {}
    for col in categorical_columns:
        cat_data[col] = data.get(col, "(欠損)")

    # race_noの正規化
    if "race_no" in cat_data:
        cat_data["race_no"] = str(cat_data["race_no"]).upper().replace("R", "")

    # カテゴリカルデータをDataFrameに変換してワンホットエンコーディング
    cat_df = pd.DataFrame([cat_data])
    cat_encoded = pd.get_dummies(cat_df, prefix=categorical_columns, dummy_na=False)

    # 数値特徴量をDataFrameに変換
    num_df = pd.DataFrame([numeric_features])

    # 結合
    X = pd.concat([cat_encoded, num_df], axis=1)

    # 訓練時の特徴量に合わせる（不足している列を追加、余分な列は削除）
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    return X


def generate_betting_strategy(probability: float, input_data: dict) -> dict:
    """買い方の提案を生成"""
    car_nos = [
        int(input_data.get("pos1_car_no", 0)),
        int(input_data.get("pos2_car_no", 0)),
        int(input_data.get("pos3_car_no", 0)),
    ]

    strategy = {
        "confidence": "高" if probability > 0.7 else "中" if probability > 0.5 else "低",
        "recommendations": []
    }

    if probability > 0.6:
        # 高確率で荒れる場合
        strategy["recommendations"].append({
            "type": "3連単",
            "description": "高配当が期待できます。3連単でのボックス買いを検討してください。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス"
        })
        strategy["recommendations"].append({
            "type": "3連複",
            "description": "リスクを抑えつつ配当を狙う場合は3連複もおすすめです。",
            "suggested_numbers": car_nos,
            "bet_type": "ボックス"
        })
    elif probability > 0.4:
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
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability >= 0.5)

        # 買い方の提案
        betting_strategy = generate_betting_strategy(probability, data)

        result = {
            "success": True,
            "probability": probability,
            "prediction": prediction,
            "prediction_label": "荒れる" if prediction == 1 else "堅い",
            "betting_strategy": betting_strategy,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """ヘルスチェック"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
