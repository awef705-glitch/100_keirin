import argparse
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROFILE_URL = "https://keirin.jp/sp/racerprofile"
REFERER_URL = "https://keirin.jp/sp/racersearch"

PREFECTURE_TO_REGION = {
    "北海道": "北海道",
    "青森県": "東北",
    "岩手県": "東北",
    "宮城県": "東北",
    "秋田県": "東北",
    "山形県": "東北",
    "福島県": "東北",
    "茨城県": "北関東",
    "栃木県": "北関東",
    "群馬県": "北関東",
    "埼玉県": "南関東",
    "千葉県": "南関東",
    "東京都": "南関東",
    "神奈川県": "南関東",
    "新潟県": "北陸",
    "富山県": "北陸",
    "石川県": "北陸",
    "福井県": "北陸",
    "山梨県": "中部",
    "長野県": "中部",
    "岐阜県": "中部",
    "静岡県": "東海",
    "愛知県": "東海",
    "三重県": "東海",
    "滋賀県": "近畿",
    "京都府": "近畿",
    "大阪府": "近畿",
    "兵庫県": "近畿",
    "奈良県": "近畿",
    "和歌山県": "近畿",
    "鳥取県": "中国",
    "島根県": "中国",
    "岡山県": "中国",
    "広島県": "中国",
    "山口県": "中国",
    "徳島県": "四国",
    "香川県": "四国",
    "愛媛県": "四国",
    "高知県": "四国",
    "福岡県": "九州",
    "佐賀県": "九州",
    "長崎県": "九州",
    "熊本県": "九州",
    "大分県": "九州",
    "宮崎県": "九州",
    "鹿児島県": "九州",
    "沖縄県": "九州",
}


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_float(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    cleaned = value.replace("cm", "").replace("kg", "").replace("kgf", "").replace(",", "").replace("点", "").strip()
    if cleaned.endswith("歳"):
        cleaned = cleaned[:-1]
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    num = parse_float(value)
    if num is None:
        return None
    return int(round(num))


def parse_table_rows(table) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in table.select("tr"):
        cells = [td.get_text(strip=True) for td in tr.select("td")]
        if cells:
            rows.append(cells)
    return rows


def table_to_dict(rows: List[List[str]]) -> Dict[str, str]:
    data: Dict[str, str] = {}
    i = 0
    while i + 1 < len(rows):
        headers = rows[i]
        values = rows[i + 1]
        if len(headers) == 1 and len(values) == 1:
            data[headers[0]] = values[0]
            i += 2
        elif len(headers) == len(values):
            for key, val in zip(headers, values):
                if key:
                    data[key] = val
            i += 2
        else:
            i += 1
    return data


def fetch_profile(session: requests.Session, rider_id: str) -> Optional[Dict[str, object]]:
    params = {"snum": rider_id}
    resp = session.get(PROFILE_URL, params=params, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.select("table.SJ0601_0319_table")
    if not tables:
        return None

    result: Dict[str, object] = {"rider_id": rider_id, "source_url": resp.url}

    update_tag = soup.select_one("p.al-r")
    if update_tag:
        result["profile_updated_at"] = update_tag.get_text(strip=True).replace("更新", "").strip()

    # Table 0: basic info
    basic = table_to_dict(parse_table_rows(tables[0]))
    result["name"] = basic.get("氏名")
    result["prefecture"] = basic.get("府県")
    result["period"] = basic.get("期別")
    result["age"] = parse_int(basic.get("年齢"))
    result["current_grade"] = basic.get("級班")
    result["registration_no"] = basic.get("登録番号")
    result["region"] = PREFECTURE_TO_REGION.get(result["prefecture"])

    # Table 1: detail info
    if len(tables) > 1:
        detail = table_to_dict(parse_table_rows(tables[1]))
        result["name_kana"] = detail.get("氏名 （フリガナ）")
        result["birthdate"] = detail.get("生年月日")
        result["grade_join_date"] = detail.get("級班所属日")
        result["current_official_score"] = parse_float(detail.get("今期得点"))
        result["next_grade"] = detail.get("次期級班")
        result["style_profile"] = detail.get("脚質")
        result["gender"] = detail.get("性別")

    # Table 2: grade history
    if len(tables) > 2:
        history_rows = parse_table_rows(tables[2])
        history: List[Dict[str, str]] = []
        for row in history_rows[1:]:
            if len(row) >= 2:
                history.append({"grade": row[0], "start_date": row[1]})
        result["grade_history"] = json.dumps(history, ensure_ascii=False)
        dates = [parse_date(entry["start_date"]) for entry in history if entry.get("start_date")]
        dates = [d for d in dates if d]
        if dates:
            earliest = min(dates)
            result["career_start_date"] = earliest.isoformat()
            result["experience_years"] = round((date.today() - earliest).days / 365.25, 2)

    # Table 3: physical data
    if len(tables) > 3:
        physical = table_to_dict(parse_table_rows(tables[3]))
        result["constellation"] = physical.get("星座")
        result["kyusei"] = physical.get("九星")
        result["blood_type"] = physical.get("血液型")
        result["height_cm"] = parse_float(physical.get("身長"))
        result["weight_kg"] = parse_float(physical.get("体重"))
        result["chest_cm"] = parse_float(physical.get("胸囲"))
        result["thigh_cm"] = parse_float(physical.get("太股"))
        result["back_strength_kg"] = parse_float(physical.get("背筋力"))
        result["lung_capacity"] = physical.get("肺活量")

    # Table 5: preferences & history
    if len(tables) > 5:
        misc = table_to_dict(parse_table_rows(tables[5]))
        result["home_bank"] = misc.get("ホームバンク")
        result["home_training_track"] = misc.get("ホーム競技場（練習地）")
        result["favorite_track_length"] = misc.get("得意な周長")
        result["favorite_track"] = misc.get("得意な競輪場")
        result["favorite_bike_no"] = misc.get("好きな車番")
        result["favorite_food"] = misc.get("好きな食べ物")
        result["disliked_food"] = misc.get("嫌いな食べ物")
        result["cycling_history"] = misc.get("自転車競技歴")
        result["junior_high_sport"] = misc.get("中学校スポーツ歴")
        result["high_school_sport"] = misc.get("高校スポーツ歴")
        result["college_sport"] = misc.get("大学・社会人スポーツ歴")

    # Table 6: keirin school stats
    if len(tables) > 6:
        school = table_to_dict(parse_table_rows(tables[6]))
        result["exam_category"] = school.get("受験区分")
        result["keirin_school_firsts"] = parse_int(school.get("競輪学校1着回数"))
        result["keirin_school_seconds"] = parse_int(school.get("競輪学校2着回数"))
        result["keirin_school_thirds"] = parse_int(school.get("競輪学校3着回数"))
        result["keirin_school_other_finish"] = parse_int(school.get("競輪学校着外回数"))
        result["keirin_school_rank"] = school.get("競輪学校順位")
        result["keirin_school_record"] = school.get("卒業記念レース")

    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch rider profile data from keirin.jp")
    parser.add_argument("--entries", default="data/keirin_race_detail_20240101_20240331_entries_full.csv", help="Entries CSV to derive rider IDs")
    parser.add_argument("--output", default="data/keirin_rider_profiles_20240101_20240331.csv", help="Output CSV path")
    parser.add_argument("--start-index", type=int, default=0, help="Optional start index for rider list")
    parser.add_argument("--limit", type=int, help="Optional limit for riders (for testing)")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between requests (seconds)")
    args = parser.parse_args()

    entries_path = Path(args.entries)
    if not entries_path.exists():
        raise SystemExit(f"Entries CSV not found: {entries_path}")

    df_entries = pd.read_csv(entries_path, dtype={"sensyuRegistNo": str, "sensyuName": str, "huKen": str})
    if "sensyuRegistNo" not in df_entries.columns:
        raise SystemExit("Column 'sensyuRegistNo' missing in entries CSV")

    df_entries["rider_id"] = df_entries["sensyuRegistNo"].fillna("0").astype(str).str.zfill(6)
    rider_meta = (
        df_entries[["rider_id", "sensyuName", "huKen", "kyakusitu", "heikinTokuten"]]
        .drop_duplicates(subset=["rider_id"])
        .rename(
            columns={
                "sensyuName": "entry_name",
                "huKen": "entry_prefecture",
                "kyakusitu": "entry_style",
                "heikinTokuten": "entry_recent_score",
            }
        )
    )

    rider_ids = rider_meta["rider_id"].tolist()
    rider_ids = rider_ids[args.start_index :]
    if args.limit:
        rider_ids = rider_ids[: args.limit]

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": REFERER_URL,
        }
    )
    session.get(REFERER_URL, timeout=10)

    profiles: List[Dict[str, object]] = []
    failed: List[str] = []

    for idx, rider_id in enumerate(rider_ids, start=1):
        try:
            profile = fetch_profile(session, rider_id)
            if profile:
                profiles.append(profile)
            else:
                failed.append(rider_id)
        except Exception as exc:
            failed.append(rider_id)
            print(f"{rider_id}: failed to fetch ({exc})")
        if idx % 50 == 0:
            print(f"processed {idx}/{len(rider_ids)} riders")
        time.sleep(max(args.delay, 0))

    if not profiles:
        raise SystemExit("No rider profiles fetched")

    df_profiles = pd.DataFrame(profiles)
    df_profiles = df_profiles.merge(rider_meta, on="rider_id", how="left")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_profiles.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "riders_requested": len(rider_ids),
        "profiles_collected": int(len(df_profiles)),
        "failed_riders": failed,
        "output_csv": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
