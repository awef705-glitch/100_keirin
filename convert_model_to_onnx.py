#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert LightGBM model to ONNX format for browser deployment.

This script converts the trained LightGBM model to ONNX format so it can
be run in the browser using ONNX Runtime Web.
"""

import json
import lightgbm as lgb
import numpy as np
from pathlib import Path

try:
    import onnx
    import onnxmltools
    from onnxmltools.convert import convert_lightgbm
    from onnxconverter_common.data_types import FloatTensorType
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install onnx onnxmltools onnxconverter_common")
    exit(1)


def main():
    # Paths
    model_dir = Path("analysis/model_outputs")
    model_path = model_dir / "prerace_model_lgbm.txt"
    metadata_path = model_dir / "prerace_model_metadata.json"

    output_dir = Path("docs/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    feature_names_path = output_dir / "feature_names.json"

    # Load LightGBM model
    print(f"Loading LightGBM model from {model_path}...")
    booster = lgb.Booster(model_file=str(model_path))

    # Load metadata to get feature information
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    feature_names = booster.feature_name()
    num_features = len(feature_names)

    print(f"Model has {num_features} features")

    # Define input type for ONNX conversion
    # Shape: (batch_size, num_features) - batch_size can be None for dynamic
    initial_type = [('input', FloatTensorType([None, num_features]))]

    # Convert to ONNX
    print("Converting to ONNX format...")
    try:
        onnx_model = convert_lightgbm(
            booster,
            initial_types=initial_type,
            target_opset=12
        )

        # Save ONNX model
        print(f"Saving ONNX model to {onnx_path}...")
        onnx.save_model(onnx_model, str(onnx_path))

        # Save feature names for JavaScript
        feature_info = {
            "feature_names": feature_names,
            "num_features": num_features,
            "model_type": "lightgbm",
            "calibration_method": metadata.get("calibration_method", "none"),
            "input_name": "input",
            "output_name": "probabilities"
        }

        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2)

        print(f"✅ Conversion successful!")
        print(f"   - ONNX model: {onnx_path}")
        print(f"   - Feature info: {feature_names_path}")
        print(f"   - Model size: {onnx_path.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\nTrying alternative conversion method...")

        # Alternative: Export as sklearn-compatible model
        # (LightGBM doesn't always convert cleanly to ONNX)
        print("Note: You may need to use the JavaScript port of LightGBM instead")
        print("See: https://github.com/Microsoft/LightGBM/tree/master/js")

        raise


if __name__ == "__main__":
    main()
