# 競輪予測アプリ

iPhoneで競輪レースの高配当の可能性を予測し、買い方を提案するWebアプリです。

## 機能

- レース情報（場名、グレード、カテゴリー、着順予想）を入力
- 機械学習モデルによる高配当（3連単10,000円以上）の確率を予測
- 確率に基づいた買い方の提案（3連単、3連複、2連複など）
- iPhone向けレスポンシブデザイン
- PWA対応でホーム画面に追加可能

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. モデルの学習

既存のデータを使ってモデルを学習します：

```bash
cd backend
python train_model.py
```

これにより、`backend/models/` ディレクトリに以下のファイルが作成されます：
- `model.pkl` - 学習済みモデル
- `feature_names.json` - 特徴量名
- `standardization_params.json` - 標準化パラメータ
- `categorical_columns.json` - カテゴリカル列
- `metrics.json` - モデル評価指標
- `reference_data.json` - 場名・グレード・カテゴリーのリスト

### 3. サーバーの起動

```bash
python backend/app.py
```

サーバーが起動したら、ブラウザで以下にアクセス：
```
http://localhost:5000
```

## iPhoneでの使い方

### 基本的な使い方

1. **場名を選択**
   - プルダウンメニューから競輪場を選択

2. **グレードを選択**
   - F1、F2、G1、G2、G3などを選択

3. **カテゴリーを選択**
   - A級予選、S級決勝などを選択

4. **レース番号を入力**
   - 例: 1、2、3...12

5. **着順予想を入力**
   - 1着、2着、3着の車番（必須）
   - 選手名（任意）
   - 決まり手（任意）

6. **「予測する」ボタンをタップ**

7. **結果を確認**
   - 高配当の確率（%）
   - 荒れる/堅いの判定
   - 買い方の提案

### PWAとしてインストール

iPhone Safariでアプリを開き、以下の手順でホーム画面に追加できます：

1. Safari下部の「共有」ボタンをタップ
2. 「ホーム画面に追加」を選択
3. アプリ名を確認して「追加」をタップ

これにより、アプリアイコンがホーム画面に追加され、ネイティブアプリのように使用できます。

## 予測の仕組み

### モデル

- **アルゴリズム**: ロジスティック回帰
- **学習データ**: 2024年1月〜10月の約48,000レース
- **予測対象**: 3連単配当10,000円以上のレース
- **精度**: AUC 0.74、精度 70.5%

### 入力特徴量

1. **車番統計量**
   - 合計、標準偏差、範囲、中央値、最小値、最大値

2. **カテゴリカル特徴**
   - 場名、グレード、レース種別
   - 1〜3着の選手名、決まり手

### 買い方の提案ロジック

#### 確率 > 60% (高)
- **3連単ボックス**: 高配当が期待できる
- **3連複ボックス**: リスクを抑えつつ配当を狙う

#### 確率 40-60% (中)
- **2連複フォーメーション**: 的中率重視
- **3連複ボックス**: やや配当を狙う

#### 確率 < 40% (低)
- **2連複**: 堅いレース、的中率重視
- **ワイド**: 確実性を求める

## ディレクトリ構成

```
100_keirin/
├── backend/
│   ├── app.py                      # Flask APIサーバー
│   ├── train_model.py              # モデル学習スクリプト
│   └── models/                     # 学習済みモデルとパラメータ
│       ├── model.pkl
│       ├── feature_names.json
│       ├── standardization_params.json
│       ├── categorical_columns.json
│       ├── metrics.json
│       └── reference_data.json
├── frontend/
│   ├── index.html                  # メインHTML
│   ├── style.css                   # スタイルシート
│   ├── app.js                      # JavaScript
│   ├── manifest.json               # PWAマニフェスト
│   └── service-worker.js           # サービスワーカー
├── data/                           # レース結果データ
├── scripts/                        # データ収集スクリプト
├── analysis/                       # 分析スクリプト
├── requirements.txt                # Python依存関係
└── README_APP.md                   # このファイル
```

## API仕様

### GET /api/reference-data

リファレンスデータ（場名、グレード、カテゴリー）を取得

**レスポンス:**
```json
{
  "tracks": ["京王閣", "立川", ...],
  "grades": ["F1", "F2", "G1", ...],
  "categories": ["A級予選", "S級決勝", ...]
}
```

### POST /api/predict

レース情報から予測を実行

**リクエスト:**
```json
{
  "track": "京王閣",
  "grade": "F1",
  "category": "A級予選",
  "race_no": "1",
  "pos1_car_no": 2,
  "pos1_name": "山田 太郎",
  "pos1_decision": "捲り",
  "pos2_car_no": 5,
  "pos2_name": "佐藤 次郎",
  "pos2_decision": "マーク",
  "pos3_car_no": 3,
  "pos3_name": "鈴木 三郎",
  "pos3_decision": ""
}
```

**レスポンス:**
```json
{
  "success": true,
  "probability": 0.65,
  "prediction": 1,
  "prediction_label": "荒れる",
  "betting_strategy": {
    "confidence": "高",
    "recommendations": [
      {
        "type": "3連単",
        "description": "高配当が期待できます...",
        "suggested_numbers": [2, 5, 3],
        "bet_type": "ボックス"
      }
    ]
  }
}
```

## トラブルシューティング

### モデルファイルが見つからない

```bash
# モデルを再学習
cd backend
python train_model.py
```

### サーバーが起動しない

```bash
# 依存関係を再インストール
pip install -r requirements.txt

# ポート5000が使われていないか確認
lsof -i :5000
```

### iPhoneで表示が崩れる

- Safari以外のブラウザでも試してみてください
- キャッシュをクリアしてください

## 今後の拡張案

- [ ] オフライン対応（TensorFlow.jsでブラウザ上で予測）
- [ ] レース履歴の保存機能
- [ ] 予測精度の追跡
- [ ] プッシュ通知機能
- [ ] より詳細な分析結果の表示
- [ ] ダークモード対応

## ライセンス

個人利用のみ
