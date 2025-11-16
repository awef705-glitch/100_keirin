#!/bin/bash
# テスト用の予測リクエスト

curl -s -X POST http://127.0.0.1:8000/predict \
  -F "race_date=2025-01-15" \
  -F "track=京王閣" \
  -F "keirin_cd=27" \
  -F "race_no=7" \
  -F "grade=F1" \
  -F "rider_names=選手A" \
  -F "rider_names=選手B" \
  -F "rider_names=選手C" \
  -F "rider_names=選手D" \
  -F "rider_names=選手E" \
  -F "rider_names=選手F" \
  -F "rider_names=選手G" \
  -F "rider_prefectures=東京" \
  -F "rider_prefectures=神奈川" \
  -F "rider_prefectures=埼玉" \
  -F "rider_prefectures=千葉" \
  -F "rider_prefectures=茨城" \
  -F "rider_prefectures=栃木" \
  -F "rider_prefectures=群馬" \
  -F "rider_grades=S1" \
  -F "rider_grades=S1" \
  -F "rider_grades=S2" \
  -F "rider_grades=S2" \
  -F "rider_grades=A1" \
  -F "rider_grades=A1" \
  -F "rider_grades=A2" \
  -F "rider_styles=逃" \
  -F "rider_styles=追" \
  -F "rider_styles=逃" \
  -F "rider_styles=追" \
  -F "rider_styles=両" \
  -F "rider_styles=逃" \
  -F "rider_styles=追" \
  -F "rider_scores=110.5" \
  -F "rider_scores=108.3" \
  -F "rider_scores=107.8" \
  -F "rider_scores=106.2" \
  -F "rider_scores=105.1" \
  -F "rider_scores=103.5" \
  -F "rider_scores=102.0" \
  | grep -E "(予測結果|確率|推奨|波乱)" | head -20
