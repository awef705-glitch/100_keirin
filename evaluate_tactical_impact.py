import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(str(Path.cwd()))

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None

from analysis import prerace_model, betting_suggestions

def evaluate(sample_size: int = 50):
    # Load data
    results_path = Path('data/keirin_results_20240101_20251004.csv')
    entries_path = Path('data/keirin_race_detail_entries_20240101_20251004.csv')

    # Use low_memory=False to avoid dtype warnings
    results = pd.read_csv(results_path, low_memory=False)
    entries = pd.read_csv(entries_path, low_memory=False)

    # Basic cleaning: drop duplicate columns
    results = results.loc[:, ~results.columns.duplicated()]
    entries = entries.loc[:, ~entries.columns.duplicated()]

    # Clean race_no in entries
    if entries['race_no'].dtype == object:
        entries['race_no'] = entries['race_no'].astype(str).str.replace('R', '', regex=False)
    
    entries['race_date'] = pd.to_numeric(entries['race_date'], errors='coerce')
    entries['race_no'] = pd.to_numeric(entries['race_no'], errors='coerce')
    entries = entries.dropna(subset=['race_date', 'race_no'])
    entries['race_date'] = entries['race_date'].astype(int)
    entries['race_no'] = entries['race_no'].astype(int)

    if 'keirin_cd' in entries.columns:
        entries['keirin_cd'] = entries['keirin_cd'].astype(str).str.zfill(2)

    # Clean race_no in results
    if results['race_no'].dtype == object:
        results['race_no'] = results['race_no'].astype(str).str.replace('R', '', regex=False)

    # Relaxed results filtering – keep rows with valid race identifiers only
    results['race_date'] = pd.to_numeric(results['race_date'], errors='coerce')
    results['race_no'] = pd.to_numeric(results['race_no'], errors='coerce')
    
    valid_results = results.dropna(subset=['race_date', 'race_no'])
    valid_results['race_date'] = valid_results['race_date'].astype(int)
    valid_results['race_no'] = valid_results['race_no'].astype(int)

    if len(valid_results) == 0:
        print('No valid results after filtering')
        return

    sample = valid_results.sample(min(sample_size, len(valid_results)), random_state=42)
    model = prerace_model.load_model()
    metadata = prerace_model.load_metadata()

    hits = 0
    high_hits = 0
    total_high = 0

    print(f"Evaluating {len(sample)} races...")

    print(f"Evaluating {len(sample)} races...")

    processed_count = 0
    for _, row in sample.iterrows():
        processed_count += 1
        race_date = int(row['race_date'])
        race_no = int(row['race_no'])
        track_name = row.get('track', '')
        keirin_cd = str(row.get('keirin_cd', '')).replace('.0', '').zfill(2)

        race_entries = entries[(entries['race_date'] == race_date) & (entries['race_no'] == race_no)]
        if len(race_entries) < 5:
            # print(f"Skipping race {race_date} {race_no}: insufficient entries ({len(race_entries)})")
            continue

        race_info = {
            'race_date': str(race_date),
            'track': track_name,
            'keirin_cd': keirin_cd,
            'race_no': race_no,
            'grade': '',
            'category': '',
            'riders': []
        }
        for _, r in race_entries.iterrows():
            race_info['riders'].append({
                'name': r.get('name', ''),
                'prefecture': r.get('huKen', ''),
                'grade': r.get('kyuhan', ''),

        suggestions = betting_suggestions.generate_betting_suggestions(race_info, prob, 'test')
        combos = [s['combination'] for s in suggestions['suggestions']]

        # Safely extract actual combination (may be missing)
        try:
            actual = f"{int(row['pos1_car_no'])}-{int(row['pos2_car_no'])}-{int(row['pos3_car_no'])}"
        except Exception:
            actual = None

        # Safely parse payout (missing → 0)
        try:
            payout = float(str(row['trifecta_payout']).replace(',', '').replace('円', '').strip() or 0)
        except Exception:
            payout = 0.0

        if actual and actual in combos:
            hits += 1
            if payout >= 10000:
                high_hits += 1
        
        # Debug output for first 5 races
        if processed_count <= 5:
            try:
                print(f"\n--- Race {race_date} {race_no} ---")
                print(f"Model Probability: {prob:.4f}")
                print(f"Actual: {actual} (Payout: {payout})")
                print(f"Hit: {'YES' if actual and actual in combos else 'NO'}")
                
                # Show top 5 ranked riders with their scores and tactical counts
                ranked = betting_suggestions.rank_riders(race_info['riders'])
                print("Top 5 Ranked Riders:")
                for rank, (car_no, score, r) in enumerate(ranked[:5], 1):
                    print(f"  {rank}. Car {car_no} ({r.get('name', 'Unknown')}) Score: {score:.1f} "
                          f"Avg: {r.get('avg_score', 0)} "
                          f"Nige: {r.get('nige_count', 0)} Makuri: {r.get('makuri_count', 0)} Sasi: {r.get('sasi_count', 0)}")
                
                # Show where the actual winner was ranked
                if actual:
                    winner_car = int(actual.split('-')[0])
                    winner_rank = next((i for i, (c, _, _) in enumerate(ranked) if c == winner_car), -1) + 1
                    print(f"Actual Winner Car {winner_car} Rank: {winner_rank}")
            except Exception as e:
                print(f"Error in debug print: {e}")

        if payout >= 10000:
            total_high += 1

    total = len(sample)
    print(f'Overall hit rate: {hits}/{total} ({hits/total*100:.1f}%)')
    if total_high:
        print(f'High payout hit rate: {high_hits}/{total_high} ({high_hits/total_high*100:.1f}%)')
    else:
        print('No high‑payout races in sample')

if __name__ == '__main__':
    evaluate(50)
