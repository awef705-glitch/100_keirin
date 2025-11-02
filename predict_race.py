#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
競輪レース予測ツール（事前情報のみ）

使い方:
  python predict_race.py --interactive
  python predict_race.py --file race.json

出力:
  ・高配当になる確率（0.0〜1.0）
  ・推奨アクション/信頼度
  ・特徴量に基づく解説
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from analysis import prerace_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="事前情報のみで高配当レースを推定するCLIツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
入力例 (race.json):
{
  "race_date": "20241025",
  "track": "京王閣",
  "keirin_cd": "27",
  "race_no": 7,
  "grade": "F1",
  "is_first_day": false,
  "is_final_day": true,
  "riders": [
    {"name": "山田太郎", "prefecture": "東京", "grade": "S1", "style": "逃", "avg_score": 109.2},
    {"name": "佐藤花子", "prefecture": "埼玉", "grade": "S2", "style": "追", "avg_score": 108.0}
  ]
}
        """,
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="対話モード")
    parser.add_argument("--file", "-f", type=Path, help="JSONファイルから入力")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("prediction_result.json"),
        help="結果を保存するJSONファイル",
    )
    return parser.parse_args()


def _prompt_bool(prompt: str) -> bool:
    while True:
        value = input(f"{prompt} (y/n): ").strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("y か n を入力してください。")


def _prompt_float(prompt: str) -> float | None:
    value = input(prompt).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        print("数値を入力してください。")
        return _prompt_float(prompt)


def collect_race_info_interactive() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("レース情報入力")
    print("=" * 70)

    race_info: Dict[str, Any] = {}
    race_info["race_date"] = input("レース日付 (YYYYMMDD): ").strip()
    race_info["track"] = input("開催場 (例: 京王閣): ").strip()
    race_info["keirin_cd"] = input("会場コード (2桁, 例: 27): ").strip()
    race_info["race_no"] = int(input("レース番号 (例: 7): ").strip() or 0)
    race_info["grade"] = input("グレード (例: GP/G1/G2/G3/F1/F2): ").strip().upper()
    race_info["is_first_day"] = _prompt_bool("初日ですか")
    race_info["is_second_day"] = _prompt_bool("2日目ですか")
    race_info["is_final_day"] = _prompt_bool("最終日ですか")

    race_info["meeting_day"] = input("開催日程 (1〜6日目、未入力可): ").strip()
    race_info["weather_condition"] = input("天候 (例: 晴れ/雨/曇り、未入力可): ").strip()
    race_info["track_condition"] = input("バンク状態 (例: 良/やや重/重、未入力可): ").strip()

    temperature = _prompt_float("気温 (℃、未入力可): ")
    race_info["temperature"] = "" if temperature is None else f"{temperature}"
    wind_speed = _prompt_float("風速 (m/s、未入力可): ")
    race_info["wind_speed"] = "" if wind_speed is None else f"{wind_speed}"
    race_info["wind_direction"] = input("風向 (例: 向かい風、未入力可): ").strip()
    race_info["is_night_race"] = _prompt_bool("ナイター開催ですか")
    race_info["notes"] = input("メモ (未入力可): ").strip()

    print("\n選手情報を入力します。空行のみの場合は終了します。")
    riders: List[Dict[str, Any]] = []
    index = 1
    while True:
        print(f"\n--- 選手 {index} ---")
        name = input("名前 (未入力で終了): ").strip()
        if not name:
            break
        prefecture = input("府県 (例: 東京): ").strip()
        grade = input("階級 (例: S1, S2, A1, A2, A3, L1, SS): ").strip().upper()
        print("脚質は『逃』『追』『両』のいずれかで入力してください。")
        style = input("脚質 (例: 逃): ").strip()
        avg_score = _prompt_float("得点 (例: 109.5, 未入力で欠損): ")

        riders.append(
            {
                "name": name,
                "prefecture": prefecture,
                "grade": grade,
                "style": style,
                "avg_score": avg_score,
            }
        )
        index += 1

    race_info["riders"] = riders
    return race_info


def load_race_info_from_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_prediction(race_info: Dict[str, Any]) -> Dict[str, Any]:
    metadata = prerace_model.load_metadata()
    model = prerace_model.load_model()

    bundle = prerace_model.build_manual_feature_row(race_info)
    feature_frame, summary = prerace_model.align_features(bundle, metadata["feature_columns"])
    probability = prerace_model.predict_probability(feature_frame, model, metadata)
    response = prerace_model.build_prediction_response(probability, summary, metadata)

    return {
        "race_info": race_info,
        "features": feature_frame.iloc[0].to_dict(),
        "probability": response["probability"],
        "confidence": response["confidence"],
        "recommendation": response["recommendation"],
        "threshold": response["threshold"],
        "high_threshold": response["high_threshold"],
        "reasons": response["reasons"],
        "betting_plan": response.get("betting_plan"),
        "summary": response["summary"],
    }


def display_result(result: Dict[str, Any]) -> None:
    race = result["race_info"]
    probability = result["probability"]

    print("\n" + "=" * 70)
    print("予測結果")
    print("=" * 70)
    print(f"{race.get('race_date', '????')} / {race.get('track', '会場不明')} / {race.get('race_no', '?')}R")
    print(f"グレード: {race.get('grade', '不明')} / 確率: {probability:.3f}")
    print(f"信頼度 : {result['confidence']}")
    print(f"推奨   : {result['recommendation']}")
    print("-" * 70)

    for reason in result["reasons"]:
        print(f"・{reason}")

    plan = result.get("betting_plan") or {}
    if plan:
        print("\n推奨ベットプラン")
        print(f"  リスクレベル : {plan.get('risk_level', '-')}")
        print(f"  概要         : {plan.get('plan_summary', '-')}")
        for ticket in plan.get("ticket_plan", []):
            print(f"    - {ticket.get('label')}: {ticket.get('description')}")
        print(f"  資金配分     : {plan.get('money_management', '-')}")
        print(f"  ヘッジ       : {plan.get('hedge_note', '-')}")

    print("\n特徴要約:")
    summary = result["summary"]
    weather = summary.get("weather_condition") or race.get("weather_condition") or "-"
    track_condition = summary.get("track_condition") or race.get("track_condition") or "-"
    meeting_day = summary.get("meeting_day") or race.get("meeting_day") or ""
    temp = summary.get("temperature_c")
    wind_speed = summary.get("wind_speed_mps")
    wind_dir = summary.get("wind_direction") or race.get("wind_direction") or "-"
    is_night = summary.get("is_night_race")
    meeting_display = f"{meeting_day}日目" if meeting_day else "未入力"
    print("入力コンディション")
    print(f"  天候           : {weather}")
    print(f"  バンク状態     : {track_condition}")
    print(f"  開催日程       : {meeting_display}")
    if temp is not None:
        print(f"  気温           : {temp:.1f}℃")
    elif race.get('temperature'):
        print(f"  気温           : {race.get('temperature')}℃")
    else:
        print("  気温           : 未入力")
    if wind_speed is not None:
        print(f"  風速/風向      : {wind_speed:.1f} m/s {wind_dir}")
    elif race.get('wind_speed'):
        print(f"  風速/風向      : {race.get('wind_speed')} m/s {wind_dir}")
    else:
        fallback = "未入力"
        if wind_dir and wind_dir != "-":
            fallback = f"未入力 {wind_dir}"
        print(f"  風速/風向      : {fallback}")
    print(f"  ナイター       : {'はい' if is_night or race.get('is_night_race') else 'いいえ'}")
    notes = race.get('notes')
    if notes:
        print(f"  メモ           : {notes}")

    print(f"  選手数            : {summary.get('entry_count')}")
    print(f"  得点平均          : {summary.get('score_mean', 0.0):.2f}")
    print(f"  得点レンジ        : {summary.get('score_range', 0.0):.2f}")
    print(f"  得点CV            : {summary.get('score_cv', 0.0):.2f}")
    style_ratios = summary.get("style_ratios", {})
    if style_ratios:
        print(
            "  脚質構成          : "
            + ", ".join(f"{k}:{v*100:.1f}%" for k, v in style_ratios.items())
        )
    print(f"  脚質多様性(0-1)  : {summary.get('style_diversity', 0.0):.2f}")
    print(f"  地元勢の種類数    : {summary.get('prefecture_unique_count', 0)}")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    if args.file:
        race_info = load_race_info_from_file(args.file)
    elif args.interactive:
        race_info = collect_race_info_interactive()
    else:
        print("エラー: --interactive もしくは --file を指定してください。")
        return

    result = run_prediction(race_info)
    display_result(result)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存しました: {args.output}")


if __name__ == "__main__":
    main()

