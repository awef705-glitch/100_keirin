#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競輪高配当予測 - iPhone対応Webアプリ
事前データのみで高配当レースを予測
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import our clean model inference
import sys
sys.path.insert(0, str(Path(__file__).parent))
from inference_from_clean_model import predict_race
from improved_betting_suggestions import generate_betting_suggestions

app = FastAPI(title="競輪 高配当予測ツール（事前データのみ）")

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

# Check if model is available
MODEL_READY = False
try:
    from inference_from_clean_model import load_model_and_metadata
    model, metadata, stats = load_model_and_metadata()
    MODEL_READY = True
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] ROC-AUC: {metadata['metrics']['roc_auc']:.4f}")
    print(f"[INFO] Features: {len(metadata['feature_columns'])}")
except FileNotFoundError as e:
    print(f"[ERROR] Model not found: {e}")
    print(f"[INFO] Please run: python3 train_clean_model.py")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = {
        "request": request,
        "model_ready": MODEL_READY,
    }
    return templates.TemplateResponse("index_clean.html", context)


@app.get("/api/riders")
async def get_riders() -> JSONResponse:
    """Return list of riders for autocomplete."""
    return JSONResponse(content=RIDERS_DATA)


@app.get("/api/tracks")
async def get_tracks() -> JSONResponse:
    """Return list of tracks with codes."""
    return JSONResponse(content=TRACKS_DATA)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    race_date: str = Form(...),
    track: str = Form(""),
    keirin_cd: str = Form(...),
    race_no: int = Form(...),
    grade: str = Form(""),
    category: str = Form(""),
    rider_names: List[str] = Form([]),
    rider_prefectures: List[str] = Form([]),
    rider_grades: List[str] = Form([]),
    rider_styles: List[str] = Form([]),
    rider_scores: List[str] = Form([]),
    budget: int = Form(2000),
) -> HTMLResponse:
    if not MODEL_READY:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": "モデルがまだ訓練されていません。python3 train_clean_model.py を実行してください。",
            },
        )

    # Parse race date
    race_date_digits = int(race_date.replace("-", ""))

    # Parse riders
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

        if avg_score is None:
            continue  # Skip riders without score

        riders.append(
            {
                "name": name.strip(),
                "prefecture": pref.strip(),
                "grade": grade_val.strip().upper(),
                "style": style.strip(),
                "avg_score": avg_score,
            }
        )

    if len(riders) < 3:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": f"選手が{len(riders)}名しか入力されていません。最低3名必要です（競走得点を入力してください）。",
            },
        )

    # Build race info
    race_info = {
        "race_date": race_date_digits,
        "track": track.strip(),
        "keirin_cd": keirin_cd.strip().zfill(2),
        "race_no": race_no,
        "grade": grade.strip().upper(),
        "category": category.strip(),
        "riders": riders,
    }

    # Predict using clean model
    try:
        prediction = predict_race(race_info)
    except Exception as e:
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "message": f"予測に失敗しました: {str(e)}",
            },
        )

    probability = prediction['probability']
    confidence = prediction['confidence']

    # Generate betting suggestions
    try:
        betting = generate_betting_suggestions(
            race_info=race_info,
            probability=probability,
            confidence=confidence,
            budget=budget,
        )
    except Exception as e:
        print(f"[WARN] Failed to generate betting suggestions: {e}")
        betting = {'error': str(e)}

    context = {
        "request": request,
        "race": race_info,
        "prediction": prediction,
        "probability": probability,
        "confidence": confidence,
        "riders": riders,
        "betting": betting,
        "model_ready": MODEL_READY,
    }

    return templates.TemplateResponse("result_clean.html", context)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("競輪 高配当予測 Web アプリ（事前データのみ）")
    print("=" * 70)
    print(f"モデル状態: {'✓ 準備完了' if MODEL_READY else '✗ 未訓練'}")

    if not MODEL_READY:
        print("\n⚠️  モデルが訓練されていません。")
        print("   以下のコマンドを実行してください:")
        print("   1. python3 build_clean_dataset.py")
        print("   2. python3 train_clean_model.py")
        print("   3. python3 compute_track_category_stats.py")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    local_url = f"http://127.0.0.1:{port}"

    if host in {"0.0.0.0", "127.0.0.1"}:
        print(f"\n📱 PC からアクセス: {local_url}")
    else:
        print(f"\n指定ホストで待機中: http://{host}:{port}")

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
        print("\n📱 iPhone など別端末は同じネットワークに接続し、次の URL を開いてください。")
        for ip in sorted(candidate_ips):
            print(f"   - http://{ip}:{port}")
    else:
        print("\n📱 iPhone など別端末からは、PC の IP アドレスを調べて http://<PCのIPアドレス>:{port} を開いてください。")

    print("\n終了するには Ctrl+C を押してください。")
    print("=" * 70)

    uvicorn.run(app, host=host, port=port)
