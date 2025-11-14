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

from analysis import prerace_model
from analysis import betting_suggestions


app = FastAPI(title="競輪 高配当予測ツール")

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

# Load model artefacts once at startup.
MODEL = None
METADATA = {}
MODEL_READY = False
LIGHTGBM_READY = False
USE_LIGHTGBM = os.getenv("KEIRIN_ENABLE_LIGHTGBM", "").strip().lower() in {"1", "true", "yes"}

try:
    METADATA = prerace_model.load_metadata()
    MODEL_READY = True
except FileNotFoundError:
    METADATA = {}
    MODEL_READY = False

if MODEL_READY and USE_LIGHTGBM:
    try:
        MODEL = prerace_model.load_model()
        LIGHTGBM_READY = MODEL is not None
    except FileNotFoundError:
        MODEL = None
        LIGHTGBM_READY = False
    except Exception:
        MODEL = None
        LIGHTGBM_READY = False


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

    bundle = prerace_model.build_manual_feature_row(race_info)
    feature_frame, summary = prerace_model.align_features(bundle, METADATA["feature_columns"])
    model_for_inference = MODEL if LIGHTGBM_READY else None

    # Prepare race context for track/category adjustment
    race_context = {
        'track': race_info.get('track', ''),
        'category': summary.get('category', ''),
    }

    probability = prerace_model.predict_probability(feature_frame, model_for_inference, METADATA, race_context)
    result = prerace_model.build_prediction_response(probability, summary, METADATA)

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
