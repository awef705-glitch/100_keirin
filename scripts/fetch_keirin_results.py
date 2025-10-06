import argparse
import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import requests

BASE_URL = "https://keirin.jp"
SP_TOP = f"{BASE_URL}/sp/"
JSON_ENDPOINT = f"{BASE_URL}/sp/json"
RACELIST_ENDPOINT = f"{BASE_URL}/sp/racelist"

RACE_JSON_PATTERN = re.compile(r"jsonData\['SJ0306'\]\s*=\s*(\{.*?\});", re.DOTALL)


def daterange(start: datetime, end: datetime) -> Iterable[str]:
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def fetch_meetings(session: requests.Session, race_date: str):
    params = {"type": "JSJ058", "kday": race_date}
    resp = session.get(JSON_ENDPOINT, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("kInfo", [])


def extract_race_json(html: str):
    match = RACE_JSON_PATTERN.search(html)
    if not match:
        raise ValueError("Could not locate jsonData['SJ0306'] block in racelist response")
    json_text = match.group(1)
    return json.loads(json_text)


def fetch_race_results(session: requests.Session, encp: str):
    payload = {"encp": encp, "disp": "SJ0306"}
    resp = session.post(RACELIST_ENDPOINT, data=payload, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return extract_race_json(resp.text)


def normalize_race_records(meeting_meta: dict, race_json: dict, default_date: str):
    records = []
    base_record = {
        "race_date": race_json.get("kaisaiDate") or default_date,
        "track": meeting_meta.get("jyoName"),
        "grade": meeting_meta.get("gradeIconChar"),
        "meeting_icon": meeting_meta.get("kaisaiIconChar"),
        "kaisai_name": race_json.get("kaisaiName") or meeting_meta.get("kaisaiName"),
        "keirin_cd": race_json.get("keirinCd") or meeting_meta.get("KeirinCd"),
    }
    races = race_json.get("resultList") or race_json.get("raceList") or []
    for race in races:
        record = base_record.copy()
        record.update({
            "race_no": race.get("rclblRaceNo"),
            "category": race.get("rclblSyumokuName"),
        })
        for idx, list_key in enumerate(("tyakui1List", "tyakui2List", "tyakui3List"), start=1):
            fin_list = race.get(list_key) or []
            if fin_list:
                finisher = fin_list[0]
                record[f"pos{idx}_name"] = finisher.get("rclblSensyuName")
                record[f"pos{idx}_car_no"] = finisher.get("rclblSyaban")
                record[f"pos{idx}_decision"] = finisher.get("rclblKimari")
                record[f"pos{idx}_region"] = finisher.get("tiku")
        harai2 = race.get("harai2syaList") or []
        if harai2:
            record["quinella"] = harai2[0].get("kumi")
            record["quinella_popularity"] = harai2[0].get("ninki")
            record["quinella_payout"] = harai2[0].get("kingaku")
        harai3 = race.get("harai3renList") or []
        if harai3:
            record["trifecta"] = harai3[0].get("kumi")
            record["trifecta_popularity"] = harai3[0].get("ninki")
            record["trifecta_payout"] = harai3[0].get("kingaku")
        records.append(record)
    return records


def write_outputs(records, race_date: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"keirin_results_{race_date}.json"
    csv_path = out_dir / f"keirin_results_{race_date}.csv"

    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(records, f_json, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for rec in records for key in rec.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return {"json": str(json_path), "csv": str(csv_path), "fields": fieldnames}


def scrape_date(session: requests.Session, race_date: str, output_dir: Path):
    meetings = fetch_meetings(session, race_date)
    if not meetings:
        print(f"  warning: no meetings for {race_date}")
        return []
    all_records: List[dict] = []
    for meeting in meetings:
        encp = meeting.get("encPrm")
        if not encp:
            continue
        try:
            race_json = fetch_race_results(session, encp)
        except (requests.RequestException, ValueError) as exc:
            print(f"  warning: skip meeting encp={encp} ({exc})")
            continue
        meeting_records = normalize_race_records(meeting, race_json, race_date)
        all_records.extend(meeting_records)
    if all_records:
        write_outputs(all_records, race_date, output_dir)
    return all_records


def aggregate_output(records: List[dict], output_dir: Path, start_date: str, end_date: str):
    if not records:
        return
    json_path = output_dir / f"keirin_results_{start_date}_{end_date}.json"
    csv_path = output_dir / f"keirin_results_{start_date}_{end_date}.csv"

    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(records, f_json, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for rec in records for key in rec.keys()})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description="Scrape keirin race results from keirin.jp mobile site")
    parser.add_argument("--date", dest="race_date", help="Target race date in YYYYMMDD")
    parser.add_argument("--start-date", dest="start_date", help="Start date (YYYYMMDD) for range scraping")
    parser.add_argument("--end-date", dest="end_date", help="End date (YYYYMMDD) for range scraping")
    parser.add_argument("--output", dest="output", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    if not args.race_date and not (args.start_date and args.end_date):
        args.race_date = datetime.now().strftime("%Y%m%d")

    output_dir = Path(args.output)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": SP_TOP,
    })
    session.get(SP_TOP, timeout=15).raise_for_status()

    if args.start_date and args.end_date:
        start_dt = datetime.strptime(args.start_date, "%Y%m%d")
        end_dt = datetime.strptime(args.end_date, "%Y%m%d")
        if start_dt > end_dt:
            raise SystemExit("start-date must be before or equal to end-date")
        aggregated_records: List[dict] = []
        for day in daterange(start_dt, end_dt):
            print(f"Scraping {day}...")
            records = scrape_date(session, day, output_dir)
            print(f"  races saved: {len(records)}")
            aggregated_records.extend(records)
        aggregate_output(aggregated_records, output_dir, args.start_date, args.end_date)
        print(json.dumps({
            "start_date": args.start_date,
            "end_date": args.end_date,
            "total_races": len(aggregated_records),
            "aggregate_files": {
                "json": f"keirin_results_{args.start_date}_{args.end_date}.json",
                "csv": f"keirin_results_{args.start_date}_{args.end_date}.csv",
            }
        }, ensure_ascii=False, indent=2))
    else:
        race_date = args.race_date
        print(f"Scraping {race_date}...")
        records = scrape_date(session, race_date, output_dir)
        if not records:
            raise SystemExit(f"No race data found for {race_date}.")
        print(json.dumps({
            "date": race_date,
            "races_saved": len(records),
            "outputs": {
                "json": f"keirin_results_{race_date}.json",
                "csv": f"keirin_results_{race_date}.csv",
            }
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
