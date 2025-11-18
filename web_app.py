#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastAPI web app for the pre-race high-payout predictor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from analysis import betting_suggestions
# V5モデルを直接使用
from analysis.train_high_payout_model import add_derived_features, select_feature_columns
import lightgbm as lgb
import pandas as pd
import numpy as np


app = FastAPI(title="競輪 高配当予測ツール")


def build_v5_features(race_info: Dict[str, Any]) -> pd.DataFrame:
    """V5モデル用の特徴量を構築"""
    riders = race_info.get("riders", [])

    # 選手統計を計算
    scores = [r.get("avg_score") for r in riders if r.get("avg_score") is not None]

    # 脚質のカウント
    styles = [r.get("style", "").strip() for r in riders]
    style_counts = {
        "逃": sum(1 for s in styles if s in ["逃", "逃げ", "先", "先行"]),
        "捲": sum(1 for s in styles if s in ["捲", "まくり", "捲り"]),
        "差": sum(1 for s in styles if s in ["差", "差し"]),
        "追": sum(1 for s in styles if s in ["追", "追込", "追い込み", "マーク"]),
    }

    # 基本特徴量
    row = {
        "race_date": int(race_info.get("race_date", "20250101")),
        "keirin_cd": race_info.get("keirin_cd", "01"),
        "race_no_int": race_info.get("race_no", 1),
        "track": race_info.get("track", "unknown"),
        "category": race_info.get("grade", "A級"),
        "grade": race_info.get("grade", "A級"),

        # 選手得点統計
        "heikinTokuten_mean": np.mean(scores) if scores else 0.0,
        "heikinTokuten_std": np.std(scores) if len(scores) > 1 else 0.0,
        "heikinTokuten_cv": np.std(scores) / np.mean(scores) if scores and np.mean(scores) > 0 else 0.0,

        # 脚質カウント
        "nigeCnt_mean": style_counts["逃"] / 9.0 if riders else 0.0,
        "makuriCnt_mean": style_counts["捲"] / 9.0 if riders else 0.0,
        "sashiCnt_mean": style_counts["差"] / 9.0 if riders else 0.0,
        "tsuiCnt_mean": style_counts["追"] / 9.0 if riders else 0.0,

        # 脚質のCV（バラつき）
        "nigeCnt_cv": 0.5 if style_counts["逃"] > 0 else 0.0,
        "makuriCnt_cv": 0.5 if style_counts["捲"] > 0 else 0.0,
        "sashiCnt_cv": 0.5 if style_counts["差"] > 0 else 0.0,
        "tsuiCnt_cv": 0.5 if style_counts["追"] > 0 else 0.0,

        # ダミー値（手入力時は不明）
        "trifecta_popularity": 50,  # 中位の人気と仮定
    }

    df = pd.DataFrame([row])

    # V5の特徴量エンジニアリングを適用
    df = add_derived_features(df)

    # カテゴリカル特徴量を設定
    categorical_cols = ["track", "category", "grade"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # V5の特徴量セットを選択
    numeric_features, categorical_features = select_feature_columns(df)
    all_features = numeric_features + categorical_features

    # 存在しない列は0で埋める
    for feat in all_features:
        if feat not in df.columns:
            df[feat] = 0

    return df[all_features]

# Load master data
MASTER_DATA_DIR = Path("analysis/model_outputs")
RIDER_MASTER_PATH = MASTER_DATA_DIR / "rider_master.json"
TRACK_MASTER_PATH = MASTER_DATA_DIR / "track_master.json"

RIDERS_DATA: List[Dict[str, Any]] = []
TRACKS_DATA: List[Dict[str, Any]] = []

try:
    if RIDER_MASTER_PATH.exists():
        with open(RIDER_MASTER_PATH, 'r', encoding='utf-8') as f:
            RIDERS_DATA = json.load(f)
    print(f"[INFO] Loaded {len(RIDERS_DATA)} riders")
except Exception as e:
    print(f"[WARN] Failed to load rider master: {e}")

try:
    if TRACK_MASTER_PATH.exists():
        with open(TRACK_MASTER_PATH, 'r', encoding='utf-8') as f:
            TRACKS_DATA = json.load(f)
    print(f"[INFO] Loaded {len(TRACKS_DATA)} tracks")
except Exception as e:
    print(f"[WARN] Failed to load track master: {e}")

TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load V5 model artefacts once at startup.
MODEL = None
METADATA = {}
FEATURE_COLUMNS = []
MODEL_READY = False
LIGHTGBM_READY = False
USE_LIGHTGBM = os.getenv("KEIRIN_ENABLE_LIGHTGBM", "").strip().lower() in {"1", "true", "yes"}

V5_MODEL_PATH = Path("analysis/model_outputs/high_payout_model_lgbm.txt")
V5_METADATA_PATH = Path("analysis/model_outputs/high_payout_model_lgbm_metadata.json")

try:
    with open(V5_METADATA_PATH, 'r', encoding='utf-8') as f:
        METADATA = json.load(f)
    MODEL_READY = True
    print(f"✅ V5メタデータ読み込み成功: {METADATA.get('n_features')}特徴量, Precision@100={METADATA['metrics']['oof_precision_at_top_k']}")
except FileNotFoundError:
    METADATA = {}
    MODEL_READY = False
    print("⚠️ V5メタデータが見つかりません")

if MODEL_READY and USE_LIGHTGBM:
    try:
        MODEL = lgb.Booster(model_file=str(V5_MODEL_PATH))
        LIGHTGBM_READY = MODEL is not None
        print(f"✅ V5モデル読み込み成功")
    except FileNotFoundError:
        MODEL = None
        LIGHTGBM_READY = False
        print("⚠️ V5モデルファイルが見つかりません")
    except Exception as e:
        MODEL = None
        LIGHTGBM_READY = False
        print(f"⚠️ V5モデル読み込みエラー: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = {
        "request": request,
        "model_ready": MODEL_READY,
        "lightgbm_ready": LIGHTGBM_READY,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/api/riders")
async def get_riders() -> JSONResponse:
    """Return list of riders for autocomplete."""
    return JSONResponse(content=RIDERS_DATA)


@app.get("/api/tracks")
async def get_tracks() -> JSONResponse:
    """Return list of tracks with codes."""
    return JSONResponse(content=TRACKS_DATA)


@app.get("/api/rider/{rider_name}")
async def get_rider(rider_name: str) -> JSONResponse:
    """Return specific rider details."""
    for rider in RIDERS_DATA:
        if rider.get("name") == rider_name:
            return JSONResponse(content=rider)
    return JSONResponse(content={"error": "Rider not found"}, status_code=404)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    race_date: str = Form(...),
    track: str = Form(""),
    keirin_cd: str = Form(...),
    race_no: int = Form(...),
    grade: str = Form(""),
    category: str = Form(""),
    meeting_day: Optional[str] = Form(None),
    is_first_day: Optional[str] = Form(None),
    is_second_day: Optional[str] = Form(None),
    is_final_day: Optional[str] = Form(None),
    weather_condition: str = Form(""),
    track_condition: str = Form(""),
    temperature: Optional[str] = Form(None),
    wind_speed: Optional[str] = Form(None),
    wind_direction: str = Form(""),
    is_night_race: Optional[str] = Form(None),
    notes: str = Form(""),
    rider_names: List[str] = Form([]),
    rider_prefectures: List[str] = Form([]),
    rider_grades: List[str] = Form([]),
    rider_styles: List[str] = Form([]),
    rider_scores: List[str] = Form([]),
) -> HTMLResponse:
    if not MODEL_READY:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "モデルがまだ訓練されていません。analysis/train_prerace_lightgbm.py を実行してください。",
            },
        )

    race_date_digits = race_date.replace("-", "")
    riders = []
    for name, pref, grade_val, style, score in zip(
        rider_names, rider_prefectures, rider_grades, rider_styles, rider_scores
    ):
        if not name.strip():
            continue
        try:
            avg_score = float(score) if score else None
        except ValueError:
            avg_score = None
        riders.append(
            {
                "name": name.strip(),
                "prefecture": pref.strip(),
                "grade": grade_val.strip().upper(),
                "style": style.strip(),
                "avg_score": avg_score,
            }
        )

    race_info = {
        "race_date": race_date_digits,
        "track": track.strip(),
        "keirin_cd": keirin_cd.strip().zfill(2),
        "race_no": race_no,
        "grade": grade.strip().upper(),
        "meeting_day": (meeting_day or "").strip(),
        "is_first_day": is_first_day == "on",
        "is_second_day": is_second_day == "on",
        "is_final_day": is_final_day == "on",
        "weather_condition": weather_condition.strip(),
        "track_condition": track_condition.strip(),
        "temperature": (temperature or "").strip(),
        "wind_speed": (wind_speed or "").strip(),
        "wind_direction": wind_direction.strip(),
        "is_night_race": is_night_race == "on",
        "notes": notes.strip(),
        "riders": riders,
    }

    # V5モデルで予測
    feature_df = build_v5_features(race_info)

    if LIGHTGBM_READY and MODEL is not None:
        # V5モデルで予測
        pred_proba = MODEL.predict(feature_df)[0]
        probability = float(pred_proba)

        # 信頼度を判定
        if probability >= 0.7:
            confidence = "高"
            message = "このレースは高配当の可能性が高いです！"
        elif probability >= 0.4:
            confidence = "中"
            message = "このレースは中程度の荒れが予想されます。"
        else:
            confidence = "低"
            message = "このレースは堅い展開になりそうです。"
    else:
        # ルールベースのフォールバック
        scores = [r.get("avg_score") for r in riders if r.get("avg_score")]
        if scores:
            score_cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            probability = min(0.9, score_cv * 2)
        else:
            probability = 0.5
        confidence = "ルールベース"
        message = "V5モデルが利用できません。簡易予測を使用しています。"

    result = {
        "probability": probability,
        "confidence": confidence,
        "message": message,
        "model": "V5 (Precision@100=67%)" if LIGHTGBM_READY else "ルールベース"
    }

    summary = {
        "n_riders": len(riders),
        "avg_score": np.mean([r.get("avg_score", 0) for r in riders if r.get("avg_score")]) if riders else 0,
    }

    # 買い目提案を生成
    suggestions_data = betting_suggestions.generate_betting_suggestions(
        race_info=race_info,
        probability=probability,
        confidence=result.get('confidence', '')
    )

    context = {
        "request": request,
        "race": race_info,
        "probability": probability,
        "result": result,
        "riders": riders,
        "summary": summary,
        "lightgbm_ready": LIGHTGBM_READY,
        "betting_suggestions": suggestions_data,
    }
    return templates.TemplateResponse("result.html", context)


if __name__ == "__main__":
    print("=" * 70)
    print("競輪 高配当予測 Web アプリ")
    print("=" * 70)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    local_url = f"http://127.0.0.1:{port}"
    if host in {"0.0.0.0", "127.0.0.1"}:
        print(f"PC からアクセス: {local_url}")
    else:
        print(f"指定ホストで待機中: http://{host}:{port}")

    try:
        import socket
        hostname = socket.gethostname()
        candidate_ips = {
            addr[4][0]
            for addr in socket.getaddrinfo(hostname, None, family=socket.AF_INET)
            if addr[4][0] and not addr[4][0].startswith("127.")
        }
    except Exception:
        candidate_ips = set()

    if candidate_ips:
        print("スマホなど別端末は同じネットワークに接続し、次の URL を開いてください。")
        for ip in sorted(candidate_ips):
            print(f"  - http://{ip}:{port}")
    else:
        print("スマホなど別端末からは、PC の IP アドレスを調べて http://<PCのIPアドレス>:{port} を開いてください。")

    print("終了するには Ctrl+C を押してください。")
    print("=" * 70)
    uvicorn.run(app, host=host, port=port)
