import pandas as pd
import glob
from pathlib import Path
import numpy as np

print("=" * 80)
print("競輪データセット完全分析")
print("=" * 80)

# 1. Results データの分析
print("\n【1. レース結果データ (keirin_results)】")
results_files = glob.glob('data/keirin_results_202*.csv')
print(f"ファイル数: {len([f for f in results_files if '_' in Path(f).stem and len(Path(f).stem.split('_')) == 3])}")

# 主要な統合ファイルを読み込み
df_results = pd.concat([
    pd.read_csv('data/keirin_results_20240401_20240630.csv'),
    pd.read_csv('data/keirin_results_20240701_20240930.csv'),
    pd.read_csv('data/keirin_results_20241001_20241231.csv'),
    pd.read_csv('data/keirin_results_20250101_20250331.csv'),
    pd.read_csv('data/keirin_results_20250401_20250630.csv'),
    pd.read_csv('data/keirin_results_20250701_20250930.csv'),
], ignore_index=True)

print(f"総レース数: {len(df_results):,}")
print(f"期間: {df_results['race_date'].min()} ~ {df_results['race_date'].max()}")
print(f"\nカラム数: {len(df_results.columns)}")
print(f"カラム: {', '.join(df_results.columns.tolist())}")

# 三連単配当の分析（全返還を除外）
def parse_payout(value):
    if pd.isna(value) or '全返還' in str(value):
        return np.nan
    return float(str(value).replace(',', '').replace('円', ''))

df_results['trifecta_payout_num'] = df_results['trifecta_payout'].apply(parse_payout)

# 有効なレースのみで統計
valid_results = df_results[df_results['trifecta_payout_num'].notna()]

print(f"\n【三連単配当の統計】")
print(f"有効レース数: {len(valid_results):,} / {len(df_results):,}")
print(f"全返還等: {len(df_results) - len(valid_results):,}")
print(f"\n平均: {valid_results['trifecta_payout_num'].mean():,.0f}円")
print(f"中央値: {valid_results['trifecta_payout_num'].median():,.0f}円")
print(f"最小: {valid_results['trifecta_payout_num'].min():,.0f}円")
print(f"最大: {valid_results['trifecta_payout_num'].max():,.0f}円")
print(f"\n10,000円以上の高配当レース: {len(valid_results[valid_results['trifecta_payout_num'] >= 10000]):,} ({len(valid_results[valid_results['trifecta_payout_num'] >= 10000]) / len(valid_results) * 100:.1f}%)")
print(f"50,000円以上: {len(valid_results[valid_results['trifecta_payout_num'] >= 50000]):,} ({len(valid_results[valid_results['trifecta_payout_num'] >= 50000]) / len(valid_results) * 100:.1f}%)")
print(f"100,000円以上: {len(valid_results[valid_results['trifecta_payout_num'] >= 100000]):,} ({len(valid_results[valid_results['trifecta_payout_num'] >= 100000]) / len(valid_results) * 100:.1f}%)")

# 会場別統計
print(f"\n【会場別レース数 TOP10】")
print(valid_results['track'].value_counts().head(10))

# グレード別統計
print(f"\n【グレード別レース数】")
print(valid_results['grade'].value_counts())

# 2. Prerace データの分析
print("\n\n【2. レース前情報データ (keirin_prerace)】")
df_prerace = pd.concat([
    pd.read_csv('data/keirin_prerace_20240401_20240630.csv'),
    pd.read_csv('data/keirin_prerace_20240701_20240930.csv'),
    pd.read_csv('data/keirin_prerace_20241001_20241231.csv'),
], ignore_index=True)

print(f"総レース数: {len(df_prerace):,}")
print(f"カラム数: {len(df_prerace.columns)}")

# 3. Race Detail Entries データの分析
print("\n\n【3. 選手詳細エントリデータ (keirin_race_detail_entries)】")
df_entries = pd.concat([
    pd.read_csv('data/keirin_race_detail_entries_20240401_20240630.csv'),
    pd.read_csv('data/keirin_race_detail_entries_20240701_20240930.csv'),
    pd.read_csv('data/keirin_race_detail_entries_20241001_20241231.csv'),
], ignore_index=True)

print(f"総エントリ数: {len(df_entries):,}")
print(f"ユニーク選手数: {df_entries['sensyuName'].nunique():,}")
print(f"\nカラム({len(df_entries.columns)}): {', '.join(df_entries.columns.tolist())}")

# 平均得点の統計
print(f"\n【選手平均得点の統計】")
print(f"平均: {df_entries['heikinTokuten'].mean():.2f}")
print(f"中央値: {df_entries['heikinTokuten'].median():.2f}")
print(f"最小: {df_entries['heikinTokuten'].min():.2f}")
print(f"最大: {df_entries['heikinTokuten'].max():.2f}")

# 脚質分布
print(f"\n【脚質分布】")
print(df_entries['kyakusitu'].value_counts())

# 級班分布
print(f"\n【級班分布】")
print(df_entries['kyuhan'].value_counts())

# 4. データ統合の可能性
print("\n\n【4. データ統合可能性の確認】")
print(f"Results: race_date, race_no, track でユニーク識別")
print(f"Prerace: race_date, race_no, track でユニーク識別")
print(f"Entries: race_encp (レース暗号化ID) で識別")
print(f"\n統合キー候補: race_date + track + race_no")

print("\n" + "=" * 80)
print("分析完了！")
print("=" * 80)
