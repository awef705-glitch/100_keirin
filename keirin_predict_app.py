#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
競輪V5予測アプリ - レース前予測版
これから開催されるレースの情報を入力して、高配当を予測
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import json
import sys
import os
from pathlib import Path

# V5モデルと特徴量エンジニアリングをインポート
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analysis.train_high_payout_model import add_derived_features, select_feature_columns
    import lightgbm as lgb
    import pandas as pd
    import numpy as np
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ モデル読み込みエラー: {e}")
    MODEL_AVAILABLE = False

# V5モデルを読み込み
V5_MODEL_PATH = Path("analysis/model_outputs/high_payout_model_lgbm.txt")
v5_model = None

if MODEL_AVAILABLE and V5_MODEL_PATH.exists():
    try:
        v5_model = lgb.Booster(model_file=str(V5_MODEL_PATH))
        print(f"✅ V5モデル読み込み完了")
    except Exception as e:
        print(f"⚠️ V5モデル読み込みエラー: {e}")


class PredictHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.serve_input_form()
        else:
            self.send_404()

    def do_POST(self):
        if self.path == '/predict':
            self.handle_prediction()
        else:
            self.send_404()

    def send_404(self):
        self.send_response(404)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write('<h1>404 Not Found</h1>'.encode())

    def serve_input_form(self):
        """レース情報入力フォーム"""

        html = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <title>競輪V5予測 - レース予測</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px 10px 80px 10px;
        }

        .container {
            max-width: 600px;
            margin: 0 auto;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-radius: 20px 20px 0 0;
        }

        .header h1 {
            font-size: 24px;
            color: #667eea;
            margin-bottom: 5px;
        }

        .header .subtitle {
            font-size: 14px;
            color: #666;
        }

        .form-card {
            background: white;
            padding: 25px 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
            font-weight: 600;
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #eee;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .rider-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 15px;
        }

        .rider-number {
            display: inline-block;
            background: #667eea;
            color: white;
            width: 32px;
            height: 32px;
            line-height: 32px;
            text-align: center;
            border-radius: 50%;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .submit-button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.2s;
        }

        .submit-button:active {
            transform: scale(0.98);
        }

        .help-text {
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: white;
        }

        .loading.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏁 競輪V5予測</h1>
            <div class="subtitle">レース情報を入力して高配当を予測</div>
        </div>

        <form method="POST" action="/predict" class="form-card" id="predict-form">
            <div class="section">
                <div class="section-title">📍 レース基本情報</div>

                <div class="form-group">
                    <label for="track">会場</label>
                    <select name="track" id="track" required>
                        <option value="">--- 選択してください ---</option>
                        <option value="函館">函館</option>
                        <option value="青森">青森</option>
                        <option value="いわき平">いわき平</option>
                        <option value="弥彦">弥彦</option>
                        <option value="前橋">前橋</option>
                        <option value="取手">取手</option>
                        <option value="宇都宮">宇都宮</option>
                        <option value="大宮">大宮</option>
                        <option value="西武園">西武園</option>
                        <option value="京王閣">京王閣</option>
                        <option value="立川">立川</option>
                        <option value="松戸">松戸</option>
                        <option value="千葉">千葉</option>
                        <option value="川崎">川崎</option>
                        <option value="平塚">平塚</option>
                        <option value="小田原">小田原</option>
                        <option value="伊東">伊東</option>
                        <option value="静岡">静岡</option>
                        <option value="名古屋">名古屋</option>
                        <option value="岐阜">岐阜</option>
                        <option value="大垣">大垣</option>
                        <option value="豊橋">豊橋</option>
                        <option value="富山">富山</option>
                        <option value="松阪">松阪</option>
                        <option value="四日市">四日市</option>
                        <option value="福井">福井</option>
                        <option value="奈良">奈良</option>
                        <option value="向日町">向日町</option>
                        <option value="和歌山">和歌山</option>
                        <option value="岸和田">岸和田</option>
                        <option value="玉野">玉野</option>
                        <option value="広島">広島</option>
                        <option value="防府">防府</option>
                        <option value="高松">高松</option>
                        <option value="小松島">小松島</option>
                        <option value="高知">高知</option>
                        <option value="松山">松山</option>
                        <option value="小倉">小倉</option>
                        <option value="久留米">久留米</option>
                        <option value="武雄">武雄</option>
                        <option value="佐世保">佐世保</option>
                        <option value="別府">別府</option>
                        <option value="熊本">熊本</option>
                    </select>
                </div>

                <div class="grid-2">
                    <div class="form-group">
                        <label for="category">クラス</label>
                        <select name="category" id="category" required>
                            <option value="">選択</option>
                            <option value="S1">S級1班</option>
                            <option value="S2">S級2班</option>
                            <option value="A1">A級1班</option>
                            <option value="A2">A級2班</option>
                            <option value="A3">A級3班</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="grade">グレード</label>
                        <select name="grade" id="grade">
                            <option value="一般">一般</option>
                            <option value="G3">G3</option>
                            <option value="G2">G2</option>
                            <option value="G1">G1</option>
                            <option value="GP">GP</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="section">
                <div class="section-title">🚴 選手情報（9名）</div>
                <p class="help-text">各選手の平均得点と脚質を入力してください</p>

                <div id="riders-container">
                    <!-- 選手1-9のフォーム -->
                </div>
            </div>

            <button type="submit" class="submit-button">
                🔮 高配当を予測する
            </button>

            <div class="loading" id="loading">
                <p>🤔 AI が分析中...</p>
            </div>
        </form>
    </div>

    <script>
        // 選手フォームを生成
        const ridersContainer = document.getElementById('riders-container');
        for (let i = 1; i <= 9; i++) {
            const riderCard = document.createElement('div');
            riderCard.className = 'rider-card';
            riderCard.innerHTML = `
                <div class="rider-number">${i}</div>
                <div class="grid-2">
                    <div class="form-group">
                        <label>平均得点</label>
                        <input type="number" name="score_${i}" step="0.01"
                               placeholder="例: 85.50" required>
                    </div>
                    <div class="form-group">
                        <label>脚質</label>
                        <select name="style_${i}" required>
                            <option value="逃">逃げ</option>
                            <option value="捲">まくり</option>
                            <option value="差">差し</option>
                            <option value="追">追込</option>
                        </select>
                    </div>
                </div>
            `;
            ridersContainer.appendChild(riderCard);
        }

        // フォーム送信時
        document.getElementById('predict-form').addEventListener('submit', function() {
            document.getElementById('loading').classList.add('active');
        });
    </script>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def handle_prediction(self):
        """予測処理"""

        # POSTデータを取得
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        params = parse_qs(post_data)

        # データを抽出
        track = params.get('track', [''])[0]
        category = params.get('category', [''])[0]
        grade = params.get('grade', ['一般'])[0]

        # 9選手のデータを抽出
        riders = []
        for i in range(1, 10):
            score = float(params.get(f'score_{i}', [0])[0])
            style = params.get(f'style_{i}', ['逃'])[0]
            riders.append({'score': score, 'style': style})

        if not MODEL_AVAILABLE or v5_model is None:
            self.serve_error("モデルが利用できません")
            return

        # 特徴量を構築
        try:
            # 選手データから統計を計算
            scores = [r['score'] for r in riders]
            styles = [r['style'] for r in riders]

            # 脚質のカウント
            nige_cnt = styles.count('逃')
            makuri_cnt = styles.count('捲')
            sasi_cnt = styles.count('差')
            oi_cnt = styles.count('追')

            # 基本的な特徴量
            race_data = {
                'entry_count': 9,
                'heikinTokuten_mean': np.mean(scores),
                'heikinTokuten_max': np.max(scores),
                'heikinTokuten_min': np.min(scores),
                'heikinTokuten_std': np.std(scores),
                'heikinTokuten_cv': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
                'heikinTokuten_range': np.max(scores) - np.min(scores),
                'nigeCnt_mean': nige_cnt,
                'makuriCnt_mean': makuri_cnt,
                'sasiCnt_mean': sasi_cnt,
                'backCnt_mean': oi_cnt,
                'nigeCnt_std': 0,
                'makuriCnt_std': 0,
                'sasiCnt_std': 0,
                'backCnt_std': 0,
                'nigeCnt_cv': 0,
                'makuriCnt_cv': 0,
                'sasiCnt_cv': 0,
                'backCnt_cv': 0,
                'track': track,
                'category': category,
                'grade': grade,
            }

            # DataFrameに変換
            df = pd.DataFrame([race_data])

            # 派生特徴量を追加
            df = add_derived_features(df)

            # 特徴量を選択
            feature_cols = select_feature_columns(df)
            X = df[feature_cols]

            # カテゴリカル特徴量を変換
            categorical_features = ['track', 'category', 'grade']
            for col in categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            # 予測
            probability = v5_model.predict(X)[0]

            # 結果を表示
            self.serve_result(probability, track, category, grade, riders, scores)

        except Exception as e:
            self.serve_error(f"予測エラー: {e}")

    def serve_result(self, probability, track, category, grade, riders, scores):
        """予測結果を表示"""

        # 判定
        if probability >= 0.75:
            judgment = "🔥 超狙い目！"
            judgment_class = "super-hot"
            message = "このレースは非常に荒れる可能性が高いです！高配当のチャンス！"
        elif probability >= 0.65:
            judgment = "⭐ 狙い目"
            judgment_class = "hot"
            message = "このレースは荒れる可能性があります。高配当が期待できます。"
        elif probability >= 0.55:
            judgment = "△ やや注意"
            judgment_class = "warm"
            message = "少し波乱の可能性があります。慎重に。"
        else:
            judgment = "× 見送り推奨"
            judgment_class = "cold"
            message = "このレースは堅い展開になりそうです。"

        # 買い目提案
        suggestions = self.generate_betting_suggestions(probability, riders, scores)

        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>予測結果 - 競輪V5</title>
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
            padding: 20px 10px;
        }}

        .container {{
            max-width: 600px;
            margin: 0 auto;
        }}

        .result-card {{
            background: white;
            border-radius: 20px;
            padding: 30px 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 20px;
        }}

        .score-display {{
            text-align: center;
            padding: 30px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            margin-bottom: 20px;
        }}

        .score-value {{
            font-size: 48px;
            font-weight: bold;
            margin: 15px 0;
        }}

        .score-label {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .judgment {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }}

        .judgment.super-hot {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }}

        .judgment.hot {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            color: white;
        }}

        .judgment.warm {{
            background: #ffeaa7;
            color: #d63031;
        }}

        .judgment.cold {{
            background: #dfe6e9;
            color: #636e72;
        }}

        .message {{
            font-size: 16px;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
            line-height: 1.6;
        }}

        .info-section {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}

        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e9ecef;
        }}

        .info-row:last-child {{
            border-bottom: none;
        }}

        .info-label {{
            font-weight: 600;
            color: #666;
        }}

        .info-value {{
            color: #333;
        }}

        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #667eea;
            margin: 20px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}

        .suggestion-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }}

        .suggestion-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}

        .suggestion-desc {{
            font-size: 14px;
            color: #666;
        }}

        .button-group {{
            display: grid;
            gap: 10px;
            margin-top: 20px;
        }}

        .button {{
            padding: 15px;
            border-radius: 10px;
            text-decoration: none;
            text-align: center;
            font-weight: bold;
            transition: transform 0.2s;
        }}

        .button:active {{
            transform: scale(0.98);
        }}

        .button-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .button-secondary {{
            background: white;
            color: #667eea;
            border: 2px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="result-card">
            <div class="score-display">
                <div class="score-label">高配当確率</div>
                <div class="score-value">{probability*100:.1f}%</div>
                <div class="score-label">V5モデル予測スコア: {probability:.4f}</div>
            </div>

            <div class="judgment {judgment_class}">
                {judgment}
            </div>

            <div class="message">
                {message}
            </div>

            <div class="info-section">
                <div class="info-row">
                    <span class="info-label">会場</span>
                    <span class="info-value">{track}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">クラス</span>
                    <span class="info-value">{category}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">グレード</span>
                    <span class="info-value">{grade}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">選手得点差</span>
                    <span class="info-value">{max(scores) - min(scores):.2f}点</span>
                </div>
            </div>

            <div class="section-title">💡 おすすめの買い方</div>

            {suggestions}

            <div class="button-group">
                <a href="/" class="button button-primary">← もう一度予測する</a>
            </div>
        </div>
    </div>
</body>
</html>
"""

        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def generate_betting_suggestions(self, probability, riders, scores):
        """買い目提案を生成"""

        suggestions = ""

        if probability >= 0.75:
            # 超荒れそう → 穴狙い
            suggestions += """
            <div class="suggestion-item">
                <div class="suggestion-title">🎯 三連単ボックス（穴選手中心）</div>
                <div class="suggestion-desc">実力下位の選手を軸に、波乱を狙う買い方がおすすめです。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">💰 三連単フォーメーション</div>
                <div class="suggestion-desc">1着に穴選手、2-3着は実力上位で手堅く。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">📊 推奨配分</div>
                <div class="suggestion-desc">三連単に70%、三連複に30%で分散投資。</div>
            </div>
            """
        elif probability >= 0.65:
            # 荒れそう → バランス型
            suggestions += """
            <div class="suggestion-item">
                <div class="suggestion-title">🎯 三連単フォーメーション</div>
                <div class="suggestion-desc">中堅選手を軸に、上位・下位を絡める買い方。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">💰 三連複ボックス</div>
                <div class="suggestion-desc">実力が拮抗している選手5-6名でボックス買い。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">📊 推奨配分</div>
                <div class="suggestion-desc">三連単50%、三連複40%、二車単10%。</div>
            </div>
            """
        elif probability >= 0.55:
            # やや荒れそう → 手堅め
            suggestions += """
            <div class="suggestion-item">
                <div class="suggestion-title">🎯 三連複ボックス</div>
                <div class="suggestion-desc">実力上位3-4名を中心にボックス買い。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">💰 二車複・二車単</div>
                <div class="suggestion-desc">手堅く上位2名の組み合わせ。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">📊 推奨配分</div>
                <div class="suggestion-desc">三連複60%、二車複30%、ワイド10%。</div>
            </div>
            """
        else:
            # 堅そう → 見送りor最小額
            suggestions += """
            <div class="suggestion-item">
                <div class="suggestion-title">⚠️ 見送り推奨</div>
                <div class="suggestion-desc">このレースは堅い展開が予想されます。配当妙味が少ない可能性があります。</div>
            </div>
            <div class="suggestion-item">
                <div class="suggestion-title">💡 参加する場合</div>
                <div class="suggestion-desc">実力上位1-2名の単勝・複勝で少額勝負。</div>
            </div>
            """

        return suggestions

    def serve_error(self, error_message):
        """エラー表示"""
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>エラー</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .error-card {{
            background: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            max-width: 400px;
        }}
        h1 {{ color: #e74c3c; margin-bottom: 20px; }}
        p {{ color: #666; margin-bottom: 20px; }}
        a {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="error-card">
        <h1>⚠️ エラー</h1>
        <p>{error_message}</p>
        <a href="/">← 戻る</a>
    </div>
</body>
</html>
"""
        self.send_response(500)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, format, *args):
        """ログを抑制"""
        return


def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, PredictHandler)

    print("=" * 70)
    print("🏁 競輪V5予測アプリ - レース予測版")
    print("=" * 70)
    print()
    print("✨ 機能:")
    print("  • これから開催されるレースの予測")
    print("  • 選手情報を入力して高配当確率を予測")
    print("  • AIがおすすめの買い方を提案")
    print()
    print(f"📱 PCからアクセス: http://127.0.0.1:{port}")
    print()

    try:
        import socket
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
