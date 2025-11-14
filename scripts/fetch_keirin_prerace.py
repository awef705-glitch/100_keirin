import argparse
import json
import re
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import requests

BASE_URL = "https://keirin.jp"
SP_TOP = f"{BASE_URL}/sp/"
JSON_ENDPOINT = f"{BASE_URL}/sp/json"
RACELIST_ENDPOINT = f"{BASE_URL}/sp/racelist"

SJ0305_PATTERN = re.compile(r"jsonData\['SJ0305'\]\s*=\s*(\{.*?\});", re.DOTALL)

MAX_ENTRIES = 9

# Browser-like headers to avoid 403
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://keirin.jp/sp/',
    'X-Requested-With': 'XMLHttpRequest',
}


def daterange(start: datetime, end: datetime) -> Iterable[str]:
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def fetch_meetings(session: requests.Session, race_date: str) -> List[Dict]:
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(0.5, 1.5))

    resp = session.get(JSON_ENDPOINT, params={"type": "JSJ058", "kday": race_date},
                       headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("kInfo", []) or []


def fetch_prerace(session: requests.Session, encp: str) -> Dict:
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(0.5, 1.5))

    resp = session.post(RACELIST_ENDPOINT, data={"encp": encp, "disp": "SJ0305"},
                        headers=HEADERS, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    match = SJ0305_PATTERN.search(html)
    if not match:
        raise ValueError("SJ0305 JSON not found")
    return json.loads(match.group(1))


def build_rows(date_str: str, meeting: Dict, prerace: Dict) -> List[Dict]:
    rows = []
    track = meeting.get("jyoName")
    grade = meeting.get("gradeIconChar")
    nitiji = meeting.get("nitijiIconChar")
    kaisai_icon = meeting.get("kaisaiIconChar")
    bkeirin_cd = meeting.get("bKeirinCd")
    kstart_date = meeting.get("kStartDate")
    keirin_cd = meeting.get("KeirinCd")
    sensyu_list = meeting.get("sensyuList") or []

    sensyu_names = [entry.get("sensyuName") for entry in sensyu_list]

    for race in prerace.get("rInfo", []) or []:
        row: Dict[str, object] = {
            "race_date": date_str,
            "track": track,
            "grade": grade,
            "nitiji": nitiji,
            "kaisai_icon": kaisai_icon,
            "bkeirin_cd": bkeirin_cd,
            "kstart_date": kstart_date,
            "keirin_cd": keirin_cd,
            "meeting_encp": meeting.get("encPrm"),
            "meeting_sensyu_list": ";".join(filter(None, sensyu_names)),
            "race_no": race.get("raceNo"),
            "syumoku": race.get("syumoku"),
            "den_time": race.get("denTime"),
            "start_time": race.get("stTime"),
            "narabi_flg": race.get("narabiFlg"),
            "narabi_y_cnt": race.get("narabiYCnt"),
            "seri": race.get("seri"),
            "ozz_flg": race.get("ozzFlg"),
            "vote_flg": race.get("voteFlg"),
            "result_flg": race.get("resultFlg"),
            "race_encp": race.get("encPrm"),
        }
        entries = race.get("sInfo", []) or []
        row["entry_count"] = len(entries)
        for idx in range(1, MAX_ENTRIES + 1):
            if idx <= len(entries):
                entry = entries[idx - 1]
                row[f"entry{idx}_car_no"] = entry.get("syaban")
                row[f"entry{idx}_color"] = entry.get("syabanColor")
                row[f"entry{idx}_name"] = entry.get("senName")
                row[f"entry{idx}_assen"] = entry.get("assen")
                row[f"entry{idx}_assen_color"] = entry.get("assenColor")
            else:
                row[f"entry{idx}_car_no"] = None
                row[f"entry{idx}_color"] = None
                row[f"entry{idx}_name"] = None
                row[f"entry{idx}_assen"] = None
                row[f"entry{idx}_assen_color"] = None
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fetch pre-race lineup data from keirin.jp")
    parser.add_argument("--start-date", dest="start_date", required=True, help="Start date (YYYYMMDD)")
    parser.add_argument("--end-date", dest="end_date", required=True, help="End date (YYYYMMDD)")
    parser.add_argument("--output", dest="output", default="data", help="Output directory")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start_date, "%Y%m%d")
    end_dt = datetime.strptime(args.end_date, "%Y%m%d")
    if start_dt > end_dt:
        raise SystemExit("start-date must be before or equal to end-date")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": SP_TOP,
    })
    session.get(SP_TOP, timeout=10)

    all_rows: List[Dict] = []
    for day in daterange(start_dt, end_dt):
        try:
            meetings = fetch_meetings(session, day)
        except requests.RequestException as exc:
            print(f"{day}: failed to fetch meetings ({exc})")
            continue
        if not meetings:
            print(f"{day}: no meetings")
            continue
        print(f"{day}: meetings={len(meetings)}")
        for meeting in meetings:
            encp = meeting.get("encPrm")
            if not encp:
                continue
            try:
                prerace = fetch_prerace(session, encp)
            except (requests.RequestException, ValueError) as exc:
                print(f"  meeting {encp}: skip ({exc})")
                continue
            rows = build_rows(day, meeting, prerace)
            all_rows.extend(rows)
            print(f"  races fetched: {len(rows)}")

    if not all_rows:
        print("No pre-race data collected")
        return

    import pandas as pd

    df = pd.DataFrame(all_rows)
    start_label = args.start_date
    end_label = args.end_date
    csv_path = output_dir / f"keirin_prerace_{start_label}_{end_label}.csv"
    json_path = output_dir / f"keirin_prerace_{start_label}_{end_label}.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(json_path, orient="records", force_ascii=False)

    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "records": int(len(df)),
        "columns": df.columns.tolist(),
    }
    (output_dir / f"keirin_prerace_{start_label}_{end_label}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
