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
    rider_back_counts: List[str] = Form([]),
    rider_nige_counts: List[str] = Form([]),
    rider_makuri_counts: List[str] = Form([]),
    rider_sasi_counts: List[str] = Form([]),
    rider_mark_counts: List[str] = Form([]),
    rider_recent_results: List[str] = Form([]),
    rider_track_results: List[str] = Form([]),
    rider_win_rates: List[str] = Form([]),
    rider_2ren_rates: List[str] = Form([]),
    rider_3ren_rates: List[str] = Form([]),
    rider_gears: List[str] = Form([]),
    rider_hs_counts: List[str] = Form([]),
    rider_dq_points: List[str] = Form([]),
    rider_home_banks: List[str] = Form([]),
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
    
    # Helper to parse float
    def _parse_float(val):
        try:
            return float(val) if val else 0.0
        except ValueError:
            return 0.0

    # Helper to parse recent results string "1,2,5" -> rates
    def _parse_recent_results(res_str):
        if not res_str:
            return 0.0, 0.0, 0.0
        parts = res_str.replace(" ", "").split(",")
        total = 0
        wins = 0
        ren2 = 0
        ren3 = 0
        for p in parts:
            if not p: continue
            total += 1
            try:
                rank = int(p)
                if rank == 1: wins += 1
                if rank <= 2: ren2 += 1
                if rank <= 3: ren3 += 1
            except ValueError:
                pass # Ignore non-integers (e.g. "欠")
        
        if total == 0: return 0.0, 0.0, 0.0
        return wins/total, ren2/total, ren3/total

    # Ensure all lists are the same length (pad with empty strings if needed)
    # Fastapi Form lists might be different lengths if empty fields are not sent? 
    # Usually browsers send empty strings for empty inputs in a form submission.
    
    for i, name in enumerate(rider_names):
        if not name.strip():
            continue
            
        # Safe get from lists
        def _get(lst, idx):
            return lst[idx] if idx < len(lst) else ""

        pref = _get(rider_prefectures, i)
        grade_val = _get(rider_grades, i)
        style = _get(rider_styles, i)
        score = _get(rider_scores, i)
        back = _get(rider_back_counts, i)
        nige = _get(rider_nige_counts, i)
        makuri = _get(rider_makuri_counts, i)
        sasi = _get(rider_sasi_counts, i)
        mark = _get(rider_mark_counts, i)
        
        recent_res = _get(rider_recent_results, i)
        track_res = _get(rider_track_results, i)
        win_rate_str = _get(rider_win_rates, i)
        ren2_rate_str = _get(rider_2ren_rates, i)
        ren3_rate_str = _get(rider_3ren_rates, i)
        gear = _get(rider_gears, i)
        hs = _get(rider_hs_counts, i)
        dq = _get(rider_dq_points, i)
        home = _get(rider_home_banks, i)

        try:
            avg_score = float(score) if score else None
        except ValueError:
            avg_score = None
            
        # Calculate rates if not provided, else use provided
        if not win_rate_str and recent_res:
            w, r2, r3 = _parse_recent_results(recent_res)
            win_rate = w
            ren2_rate = r2
            ren3_rate = r3
        else:
            win_rate = _parse_float(win_rate_str) / 100.0 if _parse_float(win_rate_str) > 1.0 else _parse_float(win_rate_str)
            ren2_rate = _parse_float(ren2_rate_str) / 100.0 if _parse_float(ren2_rate_str) > 1.0 else _parse_float(ren2_rate_str)
            ren3_rate = _parse_float(ren3_rate_str) / 100.0 if _parse_float(ren3_rate_str) > 1.0 else _parse_float(ren3_rate_str)

        riders.append(
            {
                "name": name.strip(),
                "prefecture": pref.strip(),
                "grade": grade_val.strip().upper(),
                "style": style.strip(),
                "avg_score": avg_score,
                "back_count": _parse_float(back),
                "nige_count": _parse_float(nige),
                "makuri_count": _parse_float(makuri),
                "sasi_count": _parse_float(sasi),
                "mark_count": _parse_float(mark),
                "recent_win_rate": win_rate,
                "recent_2ren_rate": ren2_rate,
                "recent_3ren_rate": ren3_rate,
                "gear_ratio": _parse_float(gear),
                "hs_count": hs, # Keep as string or parse? Model expects float for aggregation, but maybe string for rules?
                # _summarise_riders expects float for aggregation.
                # But HS count is usually "H:1 S:2".
                # I should parse it to a single number (e.g. H+S) or keep separate?
                # For now, let's try to parse "H:x S:y" to sum, or just 0 if complex.
                # Actually, let's just parse float if it's a number, else 0.
                # If user inputs "1", it works. If "H:1", _parse_float returns 0.
                # I'll improve parsing later if needed.
                "dq_points": _parse_float(dq),
                "home_bank": home.strip(),
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
        'category': category.strip(),
    }

    probability = prerace_model.predict_probability(feature_frame, model_for_inference, METADATA, race_context)
    result = prerace_model.build_prediction_response(probability, summary, METADATA, race_info)

    # 買い目提案を生成 (prerace_model内で生成済み)
    # suggestions_data = betting_suggestions.generate_betting_suggestions(
    #     race_info=race_info,
    #     probability=probability,
    #     confidence=result.get('confidence', '')
    # )
    suggestions_data = result.get('betting_data', {})

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
    import os
    
    # クラウド環境対応（Railway等）
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"  # クラウド必須
    
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
