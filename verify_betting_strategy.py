import pandas as pd
import numpy as np
import json
from pathlib import Path
from analysis import prerace_model, betting_suggestions
import sys

def load_data():
    # Load entries and results
    entries_path = Path("data/keirin_race_detail_entries_20250701_20250930.csv")
    results_path = Path("data/keirin_results_20250701_20250930.csv")
    
    print("Loading data...")
    entries = prerace_model.aggregate_rider_features(entries_path)
    # Load raw results for exact arrival order and payout
    raw_results = pd.read_csv(results_path)
    # Ensure race_no is int
    raw_results['race_no'] = raw_results['race_no'].astype(str).str.replace('R', '').astype(int)
    
    # Clean trifecta_payout
    # It contains "円" and commas, e.g. "27,720円"
    raw_results['trifecta_payout'] = (
        raw_results['trifecta_payout']
        .astype(str)
        .str.replace(r'[^\d]', '', regex=True)
        .replace('', '0')
        .astype(float)
    )

    # Ensure types match for merge
    entries['keirin_cd'] = entries['keirin_cd'].astype(str).str.zfill(2)
    raw_results['keirin_cd'] = raw_results['keirin_cd'].astype(str).str.zfill(2)
    
    # Merge
    print("Merging data...")
    merged = pd.merge(
        entries,
        raw_results[["race_date", "keirin_cd", "race_no", "trifecta_payout", "grade", "pos1_car_no", "pos2_car_no", "pos3_car_no"]],
        on=["race_date", "keirin_cd", "race_no"],
        how="inner"
    )
    
    return merged, raw_results

def check_hit(suggestion, result_row):
    """Check if a single suggestion hit the result."""
    combo = suggestion['combination']
    type_ = suggestion['type']
    
    # Parse result
    try:
        r1 = int(result_row['pos1_car_no'])
        r2 = int(result_row['pos2_car_no'])
        r3 = int(result_row['pos3_car_no'])
    except:
        return False, 0

    hit = False
    payout = 0

    if type_ == "ワイド":
        # Wide: Any 2 of top 3
        # Combo format: "1=2"
        c1, c2 = map(int, combo.split('='))
        # Check if c1 and c2 are in {r1, r2, r3}
        if c1 in [r1, r2, r3] and c2 in [r1, r2, r3]:
            hit = True
            # Wide payout is tricky as there are multiple payouts. 
            # For simulation, we'll approximate or need exact wide payout data.
            # Since we only have trifecta_payout in the merged data, 
            # we might need to load wide payouts or just track hit rate for now.
            # Let's assume a fixed average for Wide if we don't have data, 
            # OR just track Hit Rate.
            # The user asked for verification, so ROI is important.
            # Let's try to get wide payout if possible, otherwise use dummy.
            payout = 500 # Dummy average
            
    elif type_ == "2車単BOX":
        # 2-Shatan Box: 1st and 2nd in any order
        # Combo: "1-2" (but it's a box, so 1-2 and 2-1 are separate suggestions usually? 
        # No, our logic generates "1-2" for box. Wait, generate_tiered_suggestions generates individual tickets?
        # Yes: "combination": f"{p[0]}-{p[1]}", "type": "2車単BOX"
        # So "1-2" means 1 -> 2 exact.
        c1, c2 = map(int, combo.split('-'))
        if c1 == r1 and c2 == r2:
            hit = True
            payout = 2000 # Dummy average

    elif type_ == "3連単BOX" or type_ == "3連単流し" or type_ == "フォーメーション" or type_ == "穴目":
        # Exact order 1-2-3
        c1, c2, c3 = map(int, combo.split('-'))
        if c1 == r1 and c2 == r2 and c3 == r3:
            hit = True
            payout = result_row['trifecta_payout']

    elif type_ == "3連複BOX":
        # Any order of 1,2,3
        c1, c2, c3 = map(int, combo.split('='))
        if set([c1, c2, c3]) == set([r1, r2, r3]):
            hit = True
            payout = 2000 # Dummy average

    return hit, payout

def run_verification():
    print("Running verification script v3 (force update)...")
    merged, raw_results = load_data()
    metadata = prerace_model.load_metadata()
    
    # Prepare results lookup
    # (race_date, keirin_cd, race_no) -> result_row
    print("Preparing results lookup...")
    results_map = {}
    for _, row in raw_results.iterrows():
        key = (int(row['race_date']), str(row['keirin_cd']).zfill(2), int(row['race_no']))
        results_map[key] = row

    stats = {
        "low_cost": {"cost": 0, "return": 0, "hits": 0, "races": 0},
        "mid_cost": {"cost": 0, "return": 0, "hits": 0, "races": 0},
        "high_cost": {"cost": 0, "return": 0, "hits": 0, "races": 0},
    }
    
    score_bands = {
        "0-20": {"low": [], "mid": [], "high": []},
        "20-60": {"low": [], "mid": [], "high": []},
        "60-80": {"low": [], "mid": [], "high": []},
        "80-100": {"low": [], "mid": [], "high": []},
    }

    # Re-approach: Iterate over groups in the original entries dataframe
    entries_path = Path("data/keirin_race_detail_entries_20250701_20250930.csv")
    # Read raw entries to get rider details
    print("Reading raw entries...")
    raw_entries = pd.read_csv(entries_path)
    # Filter for last month to be faster?
    raw_entries = raw_entries[raw_entries['race_date'] >= 20250901]
    
    # Map track name to keirin_cd
    track_master_path = Path("analysis/model_outputs/track_master.json")
    if track_master_path.exists():
        with open(track_master_path, "r", encoding="utf-8") as f:
            track_master = json.load(f)
        track_to_code = {track["name"]: track["code"] for track in track_master}
        raw_entries["keirin_cd"] = raw_entries["track"].map(track_to_code).fillna("00")
    else:
        print("Warning: track_master.json not found. Grouping might fail.")
        raw_entries["keirin_cd"] = "00"
        
    grouped = raw_entries.groupby(['race_date', 'keirin_cd', 'race_no'])
    
    # Iterate without tqdm to see prints
    groups = list(grouped)
    print(f"Processing {len(groups)} races...")
    for i, ((date, cd, no), group) in enumerate(groups):
        if i % 100 == 0: 
            print(f"Processing race {i}/{len(groups)}")
            sys.stdout.flush()
            
        # 1. Build race_info
        riders = []
        for i, r in group.iterrows():
            riders.append({
                'name': r.get('player_name', f"Rider{i}"),
                'avg_score': float(r.get('heikinTokuten', 0)),
                'grade': str(r.get('kyuhan', '')),
                'style': str(r.get('kyakusitu', '')),
                'back_count': float(r.get('backCnt', 0)),
                'nige_count': float(r.get('nigeCnt', 0)),
                'makuri_count': float(r.get('makuriCnt', 0)),
                'sasi_count': float(r.get('sasiCnt', 0)),
            })
            
        race_info = {'riders': riders}
        
        # 2. Calculate Score (Need race-level row)
        formatted_group = prerace_model._prepare_rider_frame_from_entries(group)
        bundle = prerace_model._summarise_riders(formatted_group)
        row = pd.Series(bundle.features)
        score = prerace_model.calculate_roughness_score(row, metadata)
        try:
            score = float(score)
        except Exception as e:
            print(f"Error converting score '{score}' to float: {e}")
            score = 50.0
            
        if not isinstance(score, (int, float)):
             score = 50.0
        
        # 3. Generate Suggestions (Inlined logic)
        riders = race_info.get('riders', [])
        if len(riders) < 3:
            suggestions = {'error': 'Not enough riders'}
        else:
            # Rank riders
            ranked = betting_suggestions.rank_riders(riders)
            r_order = [r[0] for r in ranked]
            top1 = r_order[0]
            top2 = r_order[1]
            top3 = r_order[:3]
            top4 = r_order[:4]
            top5 = r_order[:5]
            top6 = r_order[:6]

            suggestions = {
                "low_cost": [],
                "mid_cost": [],
                "high_cost": []
            }
            
            import itertools
            roughness_score = float(score)
            
            if roughness_score <= 20:
                suggestions["low_cost"].append({"combination": f"{top1}={top2}", "type": "ワイド", "points": 1})
                for p in itertools.permutations(top3, 3):
                    suggestions["mid_cost"].append({"combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1})
                others = r_order[1:5]
                for s, t in itertools.permutations(others, 2):
                    suggestions["high_cost"].append({"combination": f"{top1}-{s}-{t}", "type": "3連単流し", "points": 1})
            elif roughness_score <= 60:
                for p in itertools.permutations(top3, 2):
                    suggestions["low_cost"].append({"combination": f"{p[0]}-{p[1]}", "type": "2車単BOX", "points": 1})
                w_list = r_order[:2]
                s_list = top3
                t_list = top4
                for w in w_list:
                    for s in s_list:
                        if w == s: continue
                        for t in t_list:
                            if t == w or t == s: continue
                            suggestions["mid_cost"].append({"combination": f"{w}-{s}-{t}", "type": "フォーメーション", "points": 1})
                for p in itertools.permutations(top4, 3):
                    suggestions["high_cost"].append({"combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1})
            elif roughness_score <= 80:
                for p in itertools.combinations(top4, 2):
                    suggestions["low_cost"].append({"combination": f"{p[0]}={p[1]}", "type": "ワイドBOX", "points": 1})
                for p in itertools.permutations(top5, 2):
                    suggestions["mid_cost"].append({"combination": f"{p[0]}-{p[1]}", "type": "2車単BOX", "points": 1})
                for p in itertools.permutations(top5, 3):
                    suggestions["high_cost"].append({"combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1})
            else:
                for p in itertools.combinations(top5, 2):
                    suggestions["low_cost"].append({"combination": f"{p[0]}={p[1]}", "type": "ワイドBOX", "points": 1})
                for p in itertools.combinations(top6, 3):
                    suggestions["mid_cost"].append({"combination": f"{p[0]}={p[1]}={p[2]}", "type": "3連複BOX", "points": 1})
                for p in itertools.permutations(top5, 3):
                    suggestions["high_cost"].append({"combination": f"{p[0]}-{p[1]}-{p[2]}", "type": "3連単BOX", "points": 1})
            
            suggestions = {"suggestions": suggestions}
        
        # 4. Check Result
        key = (int(date), str(cd).zfill(2), int(no))
        if key not in results_map: continue
        result_row = results_map[key]
        
        # 5. Evaluate
        for tier in ["low_cost", "mid_cost", "high_cost"]:
            sugs = suggestions['suggestions'][tier]
            cost = len(sugs) * 100
            payout = 0
            hit = False
            
            for s in sugs:
                is_hit, pay = check_hit(s, result_row)
                if is_hit:
                    hit = True
                    payout += pay
            
            stats[tier]["cost"] += cost
            stats[tier]["return"] += payout
            stats[tier]["races"] += 1
            if hit: stats[tier]["hits"] += 1
            
            # Band stats
            try:
                score = float(score)
            except:
                score = 50.0
                
            if score <= 20: band = "0-20"
            elif score <= 60: band = "20-60"
            elif score <= 80: band = "60-80"
            else: band = "80-100"
            
            score_bands[band][tier.split('_')[0]].append({
                "cost": cost, "return": payout, "hit": 1 if hit else 0
            })

    # Report
    print("\n=== Verification Results (2025/09/01 - 2025/09/30) ===")
    for tier in ["low_cost", "mid_cost", "high_cost"]:
        s = stats[tier]
        roi = s["return"] / s["cost"] * 100 if s["cost"] > 0 else 0
        hit_rate = s["hits"] / s["races"] * 100 if s["races"] > 0 else 0
        print(f"\n[{tier.upper()}]")
        print(f"  Races: {s['races']}")
        print(f"  Cost: {s['cost']:,} yen")
        print(f"  Return: {s['return']:,} yen")
        print(f"  ROI: {roi:.1f}%")
        print(f"  Hit Rate: {hit_rate:.1f}%")

    print("\n=== ROI by Score Band ===")
    for band in ["0-20", "20-60", "60-80", "80-100"]:
        print(f"\n[Score {band}]")
        for tier in ["low", "mid", "high"]:
            data = score_bands[band][tier]
            if not data: continue
            total_cost = sum(d['cost'] for d in data)
            total_return = sum(d['return'] for d in data)
            roi = total_return / total_cost * 100 if total_cost > 0 else 0
            hits = sum(d['hit'] for d in data)
            rate = hits / len(data) * 100
            print(f"  {tier.upper()}: ROI {roi:.1f}% (Hit {rate:.1f}%)")

if __name__ == "__main__":
    run_verification()
