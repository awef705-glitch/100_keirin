import json
from pathlib import Path

# メタデータ更新（最適閾値を0.4に設定）
metadata_path = Path('analysis/model_outputs/prerace_model_metadata.json')
if metadata_path.exists():
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    metadata['best_threshold'] = 0.4  # 分析結果に基づく最適値
    metadata['high_confidence_threshold'] = 0.5
    metadata['threshold_analysis'] = {
        'optimal_threshold': 0.4,
        'expected_hit_rate': 0.366,
        'analysis_date': '2025-11-24'
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Updated metadata: best_threshold = 0.4 (expected hit rate: 36.6%)")
else:
    print(f"Metadata file not found: {metadata_path}")
