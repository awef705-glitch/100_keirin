#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FastAPI service for high-payout race prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from . import inference_utils

MODEL_DIR = Path("analysis") / "model_outputs"
MODEL_PATH = MODEL_DIR / "high_payout_model_lgbm.txt"
METRICS_PATH = MODEL_DIR / "high_payout_model_lgbm_metrics.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

BOOSTER = lgb.Booster(model_file=str(MODEL_PATH))

METRICS = {}
if METRICS_PATH.exists():
    METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
BAYES_THRESHOLD = METRICS.get("best_threshold_f1", 0.5)


class RiderInput(BaseModel):
    heikinTokuten: float = Field(..., description="平均得点")
    nigeCnt: float = 0
    makuriCnt: float = 0
    sasiCnt: float = 0
    markCnt: float = 0
    backCnt: float = 0
    style: Optional[str] = Field(None, description="脚質（例: 逃, 追, 両）")


class RaceInfo(BaseModel):
    race_date: int
    keirin_cd: str
    race_no: int
    track: str = ""
    grade: str = ""
    category: str = ""
    meeting_icon: str = ""
    entry_count: Optional[int] = None
    narabi_flg: float = 0
    narabi_y_cnt: float = 0
    seri: float = 0
    ozz_flg: float = 0
    vote_flg: float = 0

    @validator("keirin_cd")
    def _pad_keirin_cd(cls, value: str) -> str:
        value = value.strip()
        return value.zfill(2)


class PredictionRequest(BaseModel):
    race_info: RaceInfo
    riders: List[RiderInput]


class Recommendation(BaseModel):
    rider_index: int
    heikinTokuten: float
    style: Optional[str]
    reason: str


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    is_high_payout: bool
    recommendations: List[Recommendation]
    notes: str


app = FastAPI(title="KEIRIN High Payout Predictor", version="1.0.0")


def _prepare_features(payload: PredictionRequest):
    df, numeric_features, categorical_features = inference_utils.build_feature_dataframe(
        payload.dict()
    )

    feature_columns = numeric_features + categorical_features
    X = df[feature_columns].copy()
    for col in categorical_features:
        X[col] = X[col].astype("category")
    return X, df


def _build_recommendations(riders: List[RiderInput]) -> List[Recommendation]:
    if not riders:
        return []

    df = np.array(
        [
            (
                idx,
                rider.heikinTokuten,
                rider.style,
                rider.makuriCnt + rider.sasiCnt,
            )
            for idx, rider in enumerate(riders)
        ],
        dtype=object,
    )
    top_heikin = df[df[:, 1].argsort()[::-1]][:3]
    recs = []
    for idx, score, style, closing in top_heikin:
        reason = f"平均得点が高い（{score:.1f}）"
        if closing and closing > 0:
            reason += f"／差し・捲り合計 {closing}"
        recs.append(
            Recommendation(
                rider_index=int(idx),
                heikinTokuten=float(score),
                style=style if style else None,
                reason=reason,
            )
        )
    return recs


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.riders:
        raise HTTPException(status_code=400, detail="riders must contain at least one entry.")

    try:
        X, _ = _prepare_features(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probability = float(BOOSTER.predict(X)[0])
    is_high = probability >= BAYES_THRESHOLD
    recs = _build_recommendations(request.riders)

    notes = (
        f"基準閾値 {BAYES_THRESHOLD:.3f} に照らし、"
        + ("高配当リスクが高いと判定しました。" if is_high else "高配当リスクは低めと判定しました。")
    )

    return PredictionResponse(
        probability=probability,
        threshold=BAYES_THRESHOLD,
        is_high_payout=is_high,
        recommendations=recs,
        notes=notes,
    )
