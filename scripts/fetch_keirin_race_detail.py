import argparse
import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

BASE_URL = "https://keirin.jp"
RACE_ENDPOINT = f"{BASE_URL}/sp/race"
SP_TOP = f"{BASE_URL}/sp/"
SJ0315_PATTERN = re.compile(r"jsonData\[\"SJ0315\"\]\s*=\s*(\{.*?\});", re.DOTALL)

# Browser-like headers to avoid 403
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://keirin.jp/sp/',
}

FIELDS_RACE = [
    "syoriKbn",
    "syusouInfoExistFlg",
    "yudoSensyuName",
    "raceResult1Syaban",
    "raceResult2Syaban",
    "backCnt1Syaban",
    "backCnt2Syaban",
    "lastUpdateTime",
]

FIELDS_ENTRY = [
    "syabanBgColorInfo",
    "syabanCharColorInfo",
    "syaban",
    "wakuban",
    "sensyuRegistNo",
    "sensyuName",
    "ketujyouTuikaHojyuColorInfo",
    "ketujyouTuikaHojyu",
    "huKen",
    "prevKyuhan",
    "kyuhan",
    "kyakusitu",
    "heikinTokuten",
    "nigeCnt",
    "makuriCnt",
    "sasiCnt",
    "markCnt",
    "backCnt",
]


def fetch_race_detail(session: requests.Session, race_encp: str) -> Dict:
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(0.5, 1.5))

    resp = session.post(RACE_ENDPOINT, data={"encp": race_encp, "disp": "SJ0315"},
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    match = SJ0315_PATTERN.search(html)
    if not match:
        raise ValueError("SJ0315 JSON not found")
    data = json.loads(match.group(1))
    if data.get("resultCd") not in (0, "0", None):
        raise ValueError(f"resultCd={data.get('resultCd')}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch keirin race detail (SJ0315) for pre-race features")
    parser.add_argument("--prerace", default="data/keirin_prerace_20240101_20251004.csv", help="CSV produced by fetch_keirin_prerace")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--limit", type=int, help="Optional limit on number of races to fetch (for testing)")
    parser.add_argument("--date-start", dest="date_start", help="Filter race_date >= this (YYYYMMDD)")
    parser.add_argument("--date-end", dest="date_end", help="Filter race_date <= this (YYYYMMDD)")
    args = parser.parse_args()

    prerace_path = Path(args.prerace)
    if not prerace_path.exists():
        raise SystemExit(f"Pre-race CSV not found: {prerace_path}")

    df_prerace = pd.read_csv(prerace_path, dtype={"race_encp": str, "race_date": str})
    if "race_encp" not in df_prerace.columns:
        raise SystemExit("Column 'race_encp' not found in pre-race CSV")

    race_rows = []
    entry_rows = []

    unique_races = df_prerace.drop_duplicates(subset=["race_encp"])[
        ["race_encp", "race_date", "track", "race_no", "grade", "syumoku"]
    ]
    unique_races = unique_races[unique_races["race_encp"].notna()]

    if args.date_start:
        unique_races = unique_races[unique_races["race_date"].astype(str) >= args.date_start]
    if args.date_end:
        unique_races = unique_races[unique_races["race_date"].astype(str) <= args.date_end]

    if args.limit:
        unique_races = unique_races.head(args.limit)
    
    print(f"Total races to process: {len(unique_races)}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": SP_TOP,
    })
    session.get(SP_TOP, timeout=10)

    for idx, row in unique_races.iterrows():
        race_encp = row["race_encp"]
        if not isinstance(race_encp, str) or not race_encp:
            continue
        try:
            detail = fetch_race_detail(session, race_encp)
        except Exception as exc:
            print(f"{race_encp}: failed ({exc})")
            continue

        race_info = {"race_encp": race_encp, "race_date": row["race_date"], "track": row["track"], "race_no": row["race_no"], "grade": row["grade"], "syumoku": row["syumoku"]}
        for field in FIELDS_RACE:
            race_info[field] = detail.get(field)
        race_rows.append(race_info)

        entries = detail.get("sensyuTypeInfo") or []
        for entry in entries:
            entry_row = {"race_encp": race_encp, "race_date": row["race_date"], "track": row["track"], "race_no": row["race_no"], "grade": row["grade"], "syumoku": row["syumoku"]}
            for field in FIELDS_ENTRY:
                entry_row[field] = entry.get(field)
            entry_rows.append(entry_row)

        if len(race_rows) % 100 == 0:
            print(f"processed {len(race_rows)} races (total: {len(unique_races)})")
            time.sleep(0.5)

    if not race_rows:
        print("No race detail fetched")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_race = pd.DataFrame(race_rows)
    df_entry = pd.DataFrame(entry_rows)

    base_label = prerace_path.stem.replace("keirin_prerace_", "")
    start_label = args.date_start if args.date_start else base_label.split("_")[0]
    end_label = args.date_end if args.date_end else base_label.split("_")[-1]
    label = f"{start_label}_{end_label}"
    race_path = output_dir / f"keirin_race_detail_race_{label}.csv"
    entry_path = output_dir / f"keirin_race_detail_entries_{label}.csv"

    df_race.to_csv(race_path, index=False, encoding="utf-8-sig")
    df_entry.to_csv(entry_path, index=False, encoding="utf-8-sig")

    summary = {
        "races": int(len(df_race)),
        "entries": int(len(df_entry)),
        "race_columns": df_race.columns.tolist(),
        "entry_columns": df_entry.columns.tolist(),
        "source_prerace": str(prerace_path),
        "output_race": str(race_path),
        "output_entries": str(entry_path),
    }
    summary_path = output_dir / f"keirin_race_detail_summary_{label}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



