#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vercel serverless function for race prediction API.

This is a simplified version of the prediction logic optimized for
serverless deployment on Vercel.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os
from pathlib import Path

# Add parent directory to path to import analysis modules
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from analysis import prerace_model
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_AVAILABLE = False


class handler(BaseHTTPRequestHandler):
    """Vercel serverless function handler."""

    def do_POST(self):
        """Handle POST requests for predictions."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            # Check if model is available
            if not MODEL_AVAILABLE:
                self.send_error_response(500, "Model not available")
                return

            # Extract race information
            race_date = data.get('race_date')  # YYYYMMDD format
            race_no = data.get('race_no')
            track = data.get('track', '')
            grade = data.get('grade', '')
            category = data.get('category', '')
            riders_data = data.get('riders', [])

            # Validate input
            if not race_date or not race_no or not riders_data:
                self.send_error_response(400, "Missing required fields")
                return

            if len(riders_data) < 7:
                self.send_error_response(400, "At least 7 riders required")
                return

            # Convert riders data to DataFrame-compatible format
            riders = []
            for i, rider_data in enumerate(riders_data):
                rider = {
                    'name': rider_data.get('name', f'選手{i+1}'),
                    'heikinTokuten': float(rider_data.get('score', 0)),
                    'grade': rider_data.get('grade', ''),
                    'style': rider_data.get('style', ''),
                    'pref': rider_data.get('prefecture', ''),
                    'back_count': int(rider_data.get('back_count', 0)),
                }
                riders.append(rider)

            # Create input for prediction
            race_input = {
                'race_date': int(race_date),
                'race_no': int(race_no),
                'track': track,
                'grade': grade,
                'category': category,
                'riders': riders,
            }

            # Make prediction using prerace_model workflow
            try:
                # Load model and metadata
                metadata = prerace_model.load_metadata()
                model = prerace_model.load_model()

                # Build feature bundle from manual input
                bundle = prerace_model.build_manual_feature_row(race_input)

                # Align features to model's expected format
                feature_frame, summary = prerace_model.align_features(
                    bundle,
                    metadata["feature_columns"]
                )

                # Race context for track/category adjustment
                race_context = {
                    'track': track,
                    'category': category,
                    'race_date': race_date,
                }

                # Get roughness score (0-100)
                roughness_score = prerace_model.predict_probability(
                    feature_frame,
                    model,
                    metadata,
                    race_context
                )

                # Build full prediction response
                result = prerace_model.build_prediction_response(
                    roughness_score,
                    summary,
                    metadata,
                    race_input
                )

                # Format response for frontend
                response_data = {
                    'success': True,
                    'roughness_score': float(result.get('roughness_score', 0)),
                    'high_payout_probability': float(result.get('roughness_score', 0)) / 100.0,  # Convert to 0-1
                    'confidence': result.get('confidence', '低'),
                    'recommendation': result.get('recommendation', ''),
                    'reasons': result.get('reasons', []),
                    'suggestions': result.get('betting_plan', []),
                }

                self.send_json_response(200, response_data)

            except Exception as e:
                print(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
                self.send_error_response(500, f"Prediction failed: {str(e)}")

        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
        except Exception as e:
            print(f"Request handling error: {e}")
            import traceback
            traceback.print_exc()
            self.send_error_response(500, f"Server error: {str(e)}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_json_response(self, status_code, data):
        """Send JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def send_error_response(self, status_code, message):
        """Send error response."""
        self.send_json_response(status_code, {
            'success': False,
            'error': message
        })

    def send_cors_headers(self):
        """Send CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
