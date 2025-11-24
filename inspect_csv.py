import pandas as pd
from pathlib import Path

def inspect():
    results_path = Path('data/keirin_results_20240101_20251004.csv')
    results = pd.read_csv(results_path, nrows=5)
    print(f"Results columns: {results.columns.tolist()}")
    print(f"First 5 rows of race_no related columns:")
    for col in results.columns:
        if 'race' in col or 'no' in col:
            print(f"{col}: {results[col].tolist()}")

if __name__ == '__main__':
    inspect()
