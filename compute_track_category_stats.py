#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track/Category別の統計を計算し、特徴量として追加
"""

import json
import pandas as pd
from pathlib import Path


def main():
    print("=" * 80)
    print("Track/Category統計の計算")
    print("=" * 80)

    # Load data
    input_file = Path('data/clean_training_dataset.csv')
    df = pd.read_csv(input_file)
    print(f"\n✓ Loaded {len(df):,} races")

    # Calculate track statistics
    track_stats = df.groupby('track').agg({
        'target_high_payout': ['mean', 'count']
    }).round(4)
    track_stats.columns = ['high_payout_rate', 'race_count']
    track_stats = track_stats[track_stats['race_count'] >= 10]  # Minimum 10 races

    print(f"\n📍 Track Statistics ({len(track_stats)} tracks):")
    print(f"   Top 5 most volatile:")
    for track, row in track_stats.nlargest(5, 'high_payout_rate').iterrows():
        print(f"   {track:10s}: {row['high_payout_rate']:.1%} ({row['race_count']:.0f} races)")

    print(f"\n   Top 5 most predictable:")
    for track, row in track_stats.nsmallest(5, 'high_payout_rate').iterrows():
        print(f"   {track:10s}: {row['high_payout_rate']:.1%} ({row['race_count']:.0f} races)")

    # Calculate category statistics
    category_stats = df.groupby('category').agg({
        'target_high_payout': ['mean', 'count']
    }).round(4)
    category_stats.columns = ['high_payout_rate', 'race_count']
    category_stats = category_stats[category_stats['race_count'] >= 10]

    print(f"\n🏁 Category Statistics ({len(category_stats)} categories):")
    print(f"   Top 10 most volatile:")
    for category, row in category_stats.nlargest(10, 'high_payout_rate').iterrows():
        if category:  # Skip empty
            print(f"   {category:25s}: {row['high_payout_rate']:.1%} ({row['race_count']:.0f} races)")

    # Save as JSON for easy loading
    output_dir = Path('analysis/model_outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'track': track_stats.to_dict(orient='index'),
        'category': category_stats.to_dict(orient='index'),
        'global_high_payout_rate': float(df['target_high_payout'].mean()),
    }

    output_file = output_dir / 'track_category_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Statistics saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
