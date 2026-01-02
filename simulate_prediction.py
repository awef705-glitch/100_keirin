import sys
import os
# Add project root to sys.path
project_root = r"c:\Users\awef7\Documents\00_GitHub\00_Me\100_keirin"
if project_root not in sys.path:
    sys.path.append(project_root)

import json
from analysis import prerace_model
from analysis import betting_suggestions

def simulate():
    print("Initializing model...")
    try:
        metadata = prerace_model.load_metadata()
        model = prerace_model.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # Mock input data
    race_info = {
        "date": "20241201",
        "place": "京王閣",
        "track": "京王閣",
        "race_num": 11,
        "race_no": 11,
        "riders": [
            {"car_num": 1, "name": "佐藤太郎", "recent_win_rate": 0.0, "recent_2ren_rate": 0.0, "recent_3ren_rate": 0.0, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 2, "name": "鈴木次郎", "recent_win_rate": 0.66, "recent_2ren_rate": 1.0, "recent_3ren_rate": 1.0, "gear_ratio": 3.85, "hs_count": 8.0, "dq_points": 0.0, "home_bank": 1},
            {"car_num": 3, "name": "Rider3", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 4, "name": "Rider4", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 5, "name": "Rider5", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 6, "name": "Rider6", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 7, "name": "Rider7", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 8, "name": "Rider8", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
            {"car_num": 9, "name": "Rider9", "recent_win_rate": 0.1, "recent_2ren_rate": 0.2, "recent_3ren_rate": 0.3, "gear_ratio": 3.92, "hs_count": 0.0, "dq_points": 0.0, "home_bank": 0},
        ]
    }

    print("Running prediction...")
    try:
        # 1. Build manual features
        bundle = prerace_model.build_manual_feature_row(race_info)
        
        # 2. Align features
        feature_frame, summary = prerace_model.align_features(bundle, metadata["feature_columns"])
        
        # 3. Predict score
        score = prerace_model.predict_probability(feature_frame, model, metadata, race_info)
        
        # 4. Build response
        result = prerace_model.build_prediction_response(score, summary, metadata)
        
        print("Prediction successful.")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"FAILED during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate()
