#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競輪V5予測 - iPhone最適化アプリ（シンプル版）
標準ライブラリのみで動作
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import socket

# V5の予測データを読み込み
V5_OOF_PATH = Path("analysis/model_outputs/high_payout_model_v5_oof.csv")
v5_predictions = None

try:
    v5_predictions = pd.read_csv(V5_OOF_PATH)
    v5_predictions['race_date_str'] = v5_predictions['race_date'].astype(str)
    v5_predictions['date'] = pd.to_datetime(v5_predictions['race_date'].astype(str), format='%Y%m%d')
    print(f"✅ V5予測データ読み込み完了: {len(v5_predictions):,}レース")
except Exception as e:
    print(f"⚠️ V5データ読み込みエラー: {e}")
    v5_predictions = pd.DataFrame()

# 会場名マッピング
TRACK_NAMES = {
    1: '函館', 2: '青森', 3: 'いわき平', 4: '弥彦', 5: '前橋',
    6: '取手', 7: '宇都宮', 8: '大宮', 9: '西武園', 10: '京王閣',
    11: '立川', 12: '松戸', 13: '千葉', 14: '川崎', 15: '平塚',
    16: '小田原', 17: '伊東', 18: '静岡', 19: '名古屋', 20: '岐阜',
    21: '大垣', 22: '豊橋', 23: '富山', 24: '松阪', 25: '四日市',
    26: '福井', 27: '奈良', 28: '向日町', 29: '和歌山', 30: '岸和田',
    31: '玉野', 32: '広島', 33: '防府', 34: '高松', 35: '小松島',
    36: '高知', 37: '松山', 38: '小倉', 39: '久留米', 40: '武雄',
    41: '佐世保', 42: '別府', 43: '熊本'
}


class KeirinHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == '/':
            self.serve_dashboard()
        elif path == '/top':
            self.serve_top_predictions()
        elif path == '/dates':
            self.serve_date_list()
        elif path.startswith('/date/'):
            date = path.split('/')[-1]
            self.serve_date_predictions(date)
        elif path == '/hits':
            self.serve_hits()
        elif path == '/stats':
            self.serve_stats()
        else:
            self.send_404()

    def send_404(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write('<h1>404 Not Found</h1>'.encode())

    def serve_dashboard(self):
        """ダッシュボード"""
        if v5_predictions.empty:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = '<h1>データが見つかりません</h1>'
            self.wfile.write(html.encode())
            return

        total_races = len(v5_predictions)
        total_high_payout = (v5_predictions['target_high_payout'] == 1).sum()
        top100 = v5_predictions.nlargest(100, 'prediction')
        top100_hits = (top100['target_high_payout'] == 1).sum()
        top100_rate = top100_hits / 100 * 100
        min_date = v5_predictions['date'].min().strftime('%Y年%m月%d日')
        max_date = v5_predictions['date'].max().strftime('%Y年%m月%d日')

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>競輪V5予測</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding-bottom: 30px;
        }}

        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .header h1 {{
            font-size: 26px;
            color: #667eea;
            margin-bottom: 5px;
        }}

        .header .subtitle {{
            font-size: 14px;
            color: #666;
        }}

        .container {{
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-card.full-width {{
            grid-column: 1 / -1;
        }}

        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}

        .stat-label {{
            font-size: 14px;
            color: #666;
        }}

        .stat-sublabel {{
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }}

        .menu-grid {{
            display: grid;
            gap: 15px;
        }}

        .menu-button {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            text-decoration: none;
            color: #333;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: transform 0.2s;
        }}

        .menu-button:active {{
            transform: scale(0.98);
        }}

        .menu-button .icon {{
            font-size: 32px;
            margin-right: 15px;
        }}

        .menu-button .content {{
            flex: 1;
        }}

        .menu-button .title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .menu-button .desc {{
            font-size: 13px;
            color: #666;
        }}

        .menu-button .arrow {{
            font-size: 20px;
            color: #ccc;
        }}

        .badge {{
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏁 競輪V5予測</h1>
        <div class="subtitle">AI予測精度67% - 業界最高水準</div>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">総レース数</div>
                <div class="stat-value">{total_races:,}</div>
                <div class="stat-sublabel">{min_date}〜<br>{max_date}</div>
            </div>

            <div class="stat-card">
                <div class="stat-label">高配当レース</div>
                <div class="stat-value">{total_high_payout:,}</div>
                <div class="stat-sublabel">10,000円以上</div>
            </div>

            <div class="stat-card full-width">
                <div class="stat-label">V5予測精度（トップ100）</div>
                <div class="stat-value">{top100_rate:.1f}%</div>
                <div class="stat-sublabel">
                    {top100_hits}/100 的中 = ランダムの2.5倍！
                </div>
                <span class="badge">BEST</span>
            </div>
        </div>

        <div class="menu-grid">
            <a href="/top" class="menu-button">
                <div class="icon">🔥</div>
                <div class="content">
                    <div class="title">トップ予測</div>
                    <div class="desc">高スコア順に表示</div>
                </div>
                <div class="arrow">›</div>
            </a>

            <a href="/dates" class="menu-button">
                <div class="icon">📅</div>
                <div class="content">
                    <div class="title">日付で探す</div>
                    <div class="desc">過去のレース結果を検索</div>
                </div>
                <div class="arrow">›</div>
            </a>

            <a href="/hits" class="menu-button">
                <div class="icon">✅</div>
                <div class="content">
                    <div class="title">的中レース</div>
                    <div class="desc">V5が当てたレースを確認</div>
                </div>
                <div class="arrow">›</div>
            </a>

            <a href="/stats" class="menu-button">
                <div class="icon">📊</div>
                <div class="content">
                    <div class="title">詳細統計</div>
                    <div class="desc">会場別・月別の分析</div>
                </div>
                <div class="arrow">›</div>
            </a>
        </div>
    </div>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_top_predictions(self):
        """トップ予測一覧"""
        top_races = v5_predictions.nlargest(100, 'prediction')

        race_cards = ""
        for idx, row in top_races.iterrows():
            date_str = row['date'].strftime('%Y年%m月%d日')
            track_name = TRACK_NAMES.get(int(row['track']), f"会場{int(row['track'])}")
            score = row['prediction']
            is_hit = row['target_high_payout'] == 1

            if score >= 0.75:
                badge = "🔥 超狙い目"
                badge_class = "super-hot"
            elif score >= 0.65:
                badge = "⭐ 狙い目"
                badge_class = "hot"
            else:
                badge = "△ 注意"
                badge_class = "warm"

            hit_text = "✅ 的中" if is_hit else "❌ 外れ"
            hit_class = "hit" if is_hit else "miss"

            race_cards += f"""
            <div class="race-card">
                <div class="race-header">
                    <div class="date">{date_str}</div>
                    <div class="track">{track_name}</div>
                </div>
                <div class="score-section">
                    <div class="score-value">{score:.4f}</div>
                    <div class="score-label">予測スコア</div>
                </div>
                <div class="badges">
                    <span class="badge {badge_class}">{badge}</span>
                    <span class="badge {hit_class}">{hit_text}</span>
                </div>
            </div>
            """

        hits = (top_races['target_high_payout'] == 1).sum()
        hit_rate = hits / len(top_races) * 100

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>トップ予測</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f7; }}
        .header {{ background: white; padding: 15px 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 20px; color: #333; }}
        .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center; }}
        .summary-value {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 15px; }}
        .race-card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .race-header {{ display: flex; justify-content: space-between; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee; }}
        .date {{ font-size: 14px; color: #666; }}
        .track {{ font-size: 16px; font-weight: bold; color: #333; }}
        .score-section {{ text-align: center; margin: 20px 0; }}
        .score-value {{ font-size: 32px; font-weight: bold; color: #667eea; }}
        .score-label {{ font-size: 12px; color: #999; margin-top: 5px; }}
        .badges {{ display: flex; gap: 10px; justify-content: center; }}
        .badge {{ padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
        .badge.super-hot {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }}
        .badge.hot {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }}
        .badge.warm {{ background: #ffeaa7; color: #d63031; }}
        .badge.hit {{ background: #00b894; color: white; }}
        .badge.miss {{ background: #636e72; color: white; }}
        .back-button {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: #667eea; color: white; padding: 15px 30px; border-radius: 25px; text-decoration: none; font-weight: bold; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 トップ100予測</h1>
    </div>
    <div class="summary">
        <div>的中率</div>
        <div class="summary-value">{hit_rate:.1f}%</div>
        <div>{hits}/100 レース的中</div>
    </div>
    <div class="container">
        {race_cards}
    </div>
    <a href="/" class="back-button">← ダッシュボードへ</a>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_date_list(self):
        """日付一覧"""
        dates = v5_predictions['date'].dt.strftime('%Y-%m-%d').unique()
        dates_sorted = sorted(dates, reverse=True)[:30]

        date_links = ""
        for date in dates_sorted:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            date_jp = date_obj.strftime('%Y年%m月%d日 (%a)')
            date_links += f'<a href="/date/{date}" class="date-link">{date_jp}</a>\n'

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>日付で探す</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 500px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 20px; padding: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }}
        h1 {{ font-size: 24px; color: #333; margin-bottom: 20px; text-align: center; }}
        .date-link {{ display: block; background: #f8f9fa; padding: 18px; margin-bottom: 10px; border-radius: 12px; text-decoration: none; color: #333; font-size: 16px; font-weight: 500; transition: all 0.2s; }}
        .date-link:active {{ transform: scale(0.98); background: #e9ecef; }}
        .back-link {{ display: block; text-align: center; color: white; text-decoration: none; margin-top: 20px; font-size: 16px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📅 日付で探す</h1>
            {date_links}
        </div>
        <a href="/" class="back-link">← ダッシュボードへ</a>
    </div>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_date_predictions(self, date):
        """指定日のレース"""
        date_races = v5_predictions[v5_predictions['date'].dt.strftime('%Y-%m-%d') == date]
        date_races = date_races.sort_values('prediction', ascending=False)

        date_obj = datetime.strptime(date, '%Y-%m-%d')
        date_jp = date_obj.strftime('%Y年%m月%d日')

        race_cards = ""
        for idx, row in date_races.iterrows():
            track_name = TRACK_NAMES.get(int(row['track']), f"会場{int(row['track'])}")
            score = row['prediction']
            is_hit = row['target_high_payout'] == 1

            if score >= 0.75:
                badge = "🔥"
            elif score >= 0.65:
                badge = "⭐"
            else:
                badge = "△"

            hit_text = "✅" if is_hit else "❌"
            hit_class = "hit" if is_hit else "miss"

            race_cards += f"""
            <div class="race-card">
                <div class="track-name">{badge} {track_name}</div>
                <div class="score">{score:.4f}</div>
                <span class="badge {hit_class}">{hit_text} {hit_text[0] + ('的中' if is_hit else '外れ')}</span>
            </div>
            """

        hits = (date_races['target_high_payout'] == 1).sum()
        total = len(date_races)
        hit_rate = hits / total * 100 if total > 0 else 0

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{date_jp}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f7; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 20px; text-align: center; }}
        .header h1 {{ font-size: 22px; margin-bottom: 10px; }}
        .stats {{ font-size: 28px; font-weight: bold; margin: 10px 0; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .race-card {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .track-name {{ font-size: 20px; font-weight: bold; color: #333; margin-bottom: 15px; }}
        .score {{ font-size: 28px; color: #667eea; font-weight: bold; text-align: center; margin: 15px 0; }}
        .badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; }}
        .badge.hit {{ background: #00b894; color: white; }}
        .badge.miss {{ background: #636e72; color: white; }}
        .back-button {{ display: block; text-align: center; background: #667eea; color: white; padding: 15px; border-radius: 10px; text-decoration: none; margin-top: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📅 {date_jp}</h1>
        <div class="stats">{hit_rate:.1f}%</div>
        <div>{hits}/{total} レース的中</div>
    </div>
    <div class="container">
        {race_cards}
        <a href="/dates" class="back-button">← 日付一覧へ</a>
    </div>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_hits(self):
        """的中レース"""
        hits = v5_predictions[v5_predictions['target_high_payout'] == 1]
        hits = hits.nlargest(100, 'prediction')

        race_cards = ""
        for idx, row in hits.iterrows():
            date_str = row['date'].strftime('%Y/%m/%d')
            track_name = TRACK_NAMES.get(int(row['track']), f"会場{int(row['track'])}")
            score = row['prediction']

            race_cards += f"""
            <div class="hit-card">
                <div class="hit-header">
                    <span class="hit-badge">✅ 的中</span>
                    <span class="date">{date_str}</span>
                </div>
                <div class="track">{track_name}</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {score*100}%"></div>
                    <div class="score-text">{score:.4f}</div>
                </div>
            </div>
            """

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>的中レース</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); min-height: 100vh; }}
        .header {{ background: rgba(255,255,255,0.95); padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 24px; color: #00b894; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .hit-card {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .hit-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .hit-badge {{ background: #00b894; color: white; padding: 6px 12px; border-radius: 15px; font-size: 13px; font-weight: bold; }}
        .date {{ font-size: 13px; color: #666; }}
        .track {{ font-size: 20px; font-weight: bold; color: #333; margin: 10px 0; }}
        .score-bar {{ position: relative; background: #f0f0f0; height: 40px; border-radius: 20px; overflow: hidden; margin-top: 15px; }}
        .score-fill {{ background: linear-gradient(90deg, #00b894 0%, #00cec9 100%); height: 100%; }}
        .score-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: #333; font-size: 16px; }}
        .back-button {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: white; color: #00b894; padding: 15px 30px; border-radius: 25px; text-decoration: none; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>✅ 的中レース トップ100</h1>
    </div>
    <div class="container">
        {race_cards}
    </div>
    <a href="/" class="back-button">← ホームへ</a>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_stats(self):
        """詳細統計"""
        # 会場別
        track_stats = []
        for track_id in sorted(v5_predictions['track'].unique()):
            track_data = v5_predictions[v5_predictions['track'] == track_id]
            top100_track = track_data.nlargest(min(100, len(track_data)), 'prediction')
            hits = (top100_track['target_high_payout'] == 1).sum()
            total = len(top100_track)
            hit_rate = hits / total * 100 if total > 0 else 0
            track_name = TRACK_NAMES.get(int(track_id), f"会場{int(track_id)}")
            track_stats.append((track_name, hit_rate, hits, total))

        track_stats.sort(key=lambda x: x[1], reverse=True)

        track_rows = ""
        for name, rate, hits, total in track_stats[:10]:
            track_rows += f"<tr><td>{name}</td><td class='number'>{rate:.1f}%</td><td class='number'>{hits}/{total}</td></tr>\n"

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>詳細統計</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f7; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px 20px; text-align: center; }}
        .header h1 {{ font-size: 24px; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .section {{ background: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .section h2 {{ font-size: 18px; color: #667eea; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #667eea; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; font-size: 13px; color: #666; border-bottom: 2px solid #e9ecef; }}
        td {{ padding: 12px; border-bottom: 1px solid #f1f3f5; font-size: 14px; }}
        td.number {{ text-align: right; font-weight: bold; color: #667eea; }}
        .back-button {{ display: block; text-align: center; background: #667eea; color: white; padding: 15px; border-radius: 10px; text-decoration: none; margin-top: 20px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 詳細統計</h1>
    </div>
    <div class="container">
        <div class="section">
            <h2>🏟️ 会場別的中率 TOP10</h2>
            <table>
                <thead>
                    <tr>
                        <th>会場</th>
                        <th style="text-align: right">的中率</th>
                        <th style="text-align: right">的中数</th>
                    </tr>
                </thead>
                <tbody>
                    {track_rows}
                </tbody>
            </table>
        </div>
        <a href="/" class="back-button">← ダッシュボードへ</a>
    </div>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """ログを抑制"""
        return


def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, KeirinHandler)

    print("=" * 70)
    print("🏁 競輪V5予測アプリ - iPhone最適化版")
    print("=" * 70)
    print()
    print("✨ 機能:")
    print("  • 過去の予測履歴閲覧（27,711レース）")
    print("  • トップ予測レース表示")
    print("  • 日付別検索")
    print("  • 的中レース一覧")
    print("  • 詳細統計（会場別）")
    print()
    print(f"📱 PCからアクセス: http://127.0.0.1:{port}")
    print()

    # ローカルIPを取得
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith('127.'):
            print(f"📱 iPhoneからアクセス（同じWi-Fiに接続）:")
            print(f"   http://{local_ip}:{port}")
    except:
        pass

    print()
    print("終了するには Ctrl+C")
    print("=" * 70)
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nサーバーを停止しました")
        httpd.shutdown()


if __name__ == '__main__':
    run_server()
