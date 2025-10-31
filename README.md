# 🏁 競輪予測アプリ

iPhoneで使える競輪レース高配当予測アプリです。機械学習を使って、レースが荒れる確率を予測し、最適な買い方を提案します。

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ 特徴

- 📱 **iPhoneで完結**: すべての操作がiPhoneで可能
- 🤖 **機械学習による予測**: ロジスティック回帰モデル（AUC: 0.74）
- 💡 **買い方の提案**: 確率に基づいた券種と買い方を提案
- 🎨 **直感的なUI**: レスポンシブデザイン、入力しやすいフォーム
- 📲 **PWA対応**: ホーム画面に追加してネイティブアプリのように使用可能
- 🔒 **プライバシー保護**: ローカルネットワーク内で動作、データは外部送信なし

## 🚀 クイックスタート

### 必要なもの

- Python 3.8以上
- iPhone（iOS 11.3以上、Safari）
- 同じWi-Fiネットワーク

### インストールと起動

1. **リポジトリをクローン**
```bash
git clone https://github.com/awef705-glitch/100_keirin.git
cd 100_keirin
```

2. **依存関係をインストール**
```bash
pip install -r requirements.txt
```

3. **モデルを学習（初回のみ）**
```bash
python backend/train_model.py
```

4. **サーバーを起動**

**簡単な方法（推奨）:**
```bash
# Mac/Linux
./start.sh

# Windows
start.bat
```

**手動起動:**
```bash
python backend/app.py
```

5. **iPhoneからアクセス**

起動時に表示されるURL（例: `http://192.168.1.100:5000`）をiPhoneのSafariで開く

### 📱 iPhoneでの使い方

詳しい手順は [QUICKSTART_IPHONE.md](QUICKSTART_IPHONE.md) を参照してください。

**簡単な流れ:**
1. PCでサーバーを起動
2. iPhoneとPCを同じWi-Fiに接続
3. iPhoneのSafariで表示されたURLを開く
4. ホーム画面に追加（PWA化）
5. レース情報を入力して予測！

## 📊 予測精度

- **データセット**: 48,682レース（2024年1月〜10月）
- **高配当レース**: 12,931件（26.6%）
- **モデル性能**:
  - AUC: 0.7412
  - Average Precision: 0.5083
  - 精度: 70.5%

## 🎯 使用例

### 入力

- **基本情報**: 場名、グレード、カテゴリー、レース番号
- **着順予想（1〜3着）**: 車番、選手名（任意）、決まり手（任意）

### 出力

- **高配当の確率**: パーセンテージとプログレスバー
- **判定**: 「荒れる」または「堅い」
- **買い方の提案**:
  - 信頼度（高/中/低）
  - 推奨券種（3連単、3連複、2連複、ワイド）
  - 推奨車番と買い方

## 🏗️ プロジェクト構成

```
100_keirin/
├── backend/                    # バックエンド（Flask API）
│   ├── app.py                 # APIサーバー
│   ├── train_model.py         # モデル学習スクリプト
│   └── models/                # 学習済みモデルとパラメータ
├── frontend/                   # フロントエンド（PWA）
│   ├── index.html             # メインHTML
│   ├── style.css              # スタイルシート
│   ├── app.js                 # JavaScript
│   ├── manifest.json          # PWAマニフェスト
│   ├── service-worker.js      # サービスワーカー
│   └── icon-*.png             # アプリアイコン
├── data/                       # レース結果データ（1,294ファイル、91MB）
├── scripts/                    # ユーティリティスクリプト
│   ├── fetch_keirin_results.py    # データ収集スクリプト
│   └── generate_icons.py          # アイコン生成スクリプト
├── analysis/                   # 分析スクリプト
├── start.sh                   # 起動スクリプト（Mac/Linux）
├── start.bat                  # 起動スクリプト（Windows）
├── requirements.txt           # Python依存関係
├── README.md                  # このファイル
├── README_APP.md              # アプリ詳細仕様
└── QUICKSTART_IPHONE.md       # iPhone使用ガイド
```

## 📡 API仕様

### エンドポイント

#### `GET /api/health`
サーバーとモデルの状態を確認

**レスポンス:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `GET /api/reference-data`
場名、グレード、カテゴリーのリストを取得

**レスポンス:**
```json
{
  "tracks": ["京王閣", "立川", ...],
  "grades": ["F1", "F2", "G1", ...],
  "categories": ["A級予選", "S級決勝", ...]
}
```

#### `POST /api/predict`
レース情報から予測を実行

**リクエスト:**
```json
{
  "track": "京王閣",
  "grade": "F1",
  "category": "A級予選",
  "race_no": "1",
  "pos1_car_no": 7,
  "pos2_car_no": 9,
  "pos3_car_no": 1
}
```

**レスポンス:**
```json
{
  "success": true,
  "probability": 0.72,
  "prediction": 1,
  "prediction_label": "荒れる",
  "betting_strategy": {
    "confidence": "高",
    "recommendations": [...]
  }
}
```

## 🛠️ 開発

### データ収集

新しいレースデータを収集:
```bash
# 特定の日付
python scripts/fetch_keirin_results.py --date 20241101

# 期間指定
python scripts/fetch_keirin_results.py --start-date 20241101 --end-date 20241130
```

### モデル再学習

新しいデータでモデルを再学習:
```bash
python backend/train_model.py --input data/keirin_results_YYYYMMDD_YYYYMMDD.csv
```

### アイコン再生成

アプリアイコンを再生成:
```bash
python scripts/generate_icons.py
```

## 🐛 トラブルシューティング

### iPhoneからアクセスできない

1. **Wi-Fi接続を確認**
   - iPhoneとPCが同じWi-Fiに接続されているか確認

2. **IPアドレスを確認**
   ```bash
   # Mac/Linux
   hostname -I

   # Windows
   ipconfig
   ```

3. **ファイアウォールを確認**
   - ポート5000がブロックされていないか確認

### 予測エラー

1. **必須項目の確認**
   - 場名、グレード、カテゴリー、車番が入力されているか

2. **モデルファイルの確認**
   ```bash
   ls backend/models/
   # model.pkl が存在するか確認
   ```

3. **再学習**
   ```bash
   python backend/train_model.py
   ```

### その他の問題

詳しくは [QUICKSTART_IPHONE.md](QUICKSTART_IPHONE.md) のトラブルシューティングセクションを参照。

## 📚 ドキュメント

- [README_APP.md](README_APP.md) - アプリの詳細仕様とAPI仕様
- [QUICKSTART_IPHONE.md](QUICKSTART_IPHONE.md) - iPhoneでの使い方ガイド

## 🔒 セキュリティとプライバシー

- ローカルネットワーク内でのみ動作
- データは外部に送信されません
- 予測結果は参考値です
- 賭博行為を推奨するものではありません

## 📝 ライセンス

個人利用のみ

## 🙏 謝辞

- データソース: [keirin.jp](https://keirin.jp)
- 機械学習: scikit-learn
- WebフレームワーK: Flask

## 📮 サポート

問題や質問がある場合は、GitHubのIssuesで報告してください。

---

**注意**: このアプリは予測の精度を保証するものではありません。賭博行為はご自身の責任で行ってください。
