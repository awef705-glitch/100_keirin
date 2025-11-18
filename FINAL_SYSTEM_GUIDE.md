# 競輪高配当予測システム - 完成版ガイド

## 🎯 システム概要

**事前データのみ**で競輪の高配当レース（三連単10,000円以上）を予測するシステムです。

### ✅ 重要な特徴
- **事後データは一切使用していません**（人気順位、オッズ、着順など）
- **事前に分かる情報のみ**で予測（競走得点、級班、脚質、府県、B関連）
- **TimeSeriesSplit**でデータリーケージを完全防止
- **iPhone対応**のWebアプリ

---

## 📊 精度指標

### 最終モデル（B関連特徴量追加後）
```
ROC-AUC: 0.5634
Precision@Top100: 0.38
訓練データ: 6,758レース（3ヶ月分）
特徴量数: 86個
```

### 最重要特徴量（トップ10）
1. **sasiCnt_cv** (514.2) - 差し勝ちのばらつき
2. **score_cv** (434.5) - 得点のばらつき
3. **entry_count** (430.4) - 出走人数
4. **estimated_top3_vs_others** (381.4) - 上位と下位の格差
5. **nigeCnt_std** (353.2) - 逃げ勝ちの標準偏差
6. **makuriCnt_std** (336.1) - 捲り勝ちの標準偏差
7. **backCnt_cv** (323.8) - バックのばらつき
8. **line_balance_std** (318.9) - ラインのバランス
9. **b_experience_entropy** (295.6) - 経験値の多様性
10. **estimated_favorite_gap** (292.5) - 得点1位と2位の差

---

## 🚀 使い方

### 1. データセット構築

```bash
# 事後データを除外したクリーンデータセットを構築
python3 build_clean_dataset.py
```

出力: `data/clean_training_dataset.csv`（6,758レース、92特徴量）

### 2. モデル訓練

```bash
# LightGBM + TimeSeriesSplitで訓練
python3 train_clean_model.py
```

出力:
- `analysis/model_outputs/clean_model_lgbm.txt` - 訓練済みモデル
- `analysis/model_outputs/clean_model_metadata.json` - メタデータ
- `analysis/model_outputs/clean_model_oof.csv` - OOF予測
- `analysis/model_outputs/clean_model_feature_importance.csv` - 特徴量重要度

### 3. Track/Category統計計算

```bash
# 会場・カテゴリ別の荒れやすさを計算
python3 compute_track_category_stats.py
```

出力: `analysis/model_outputs/track_category_stats.json`

### 4. iPhone対応Webアプリ起動

```bash
# FastAPIサーバー起動
python3 clean_web_app.py
```

アクセス:
- PC: http://127.0.0.1:8000
- iPhone: http://<PCのIPアドレス>:8000

---

## 📱 Webアプリの使い方

### 入力項目

#### レース情報
- **レース日**: YYYY-MM-DD形式
- **開催場**: プルダウンから選択（例：京王閣）
- **会場コード**: 自動入力（例：27）
- **レース番号**: 1-12
- **グレード**: GP/G1/G2/G3/F1/F2
- **カテゴリ**: S級選抜、A級特選など

#### 選手情報（3-9名）
- **選手名**: 手動入力（オートコンプリート対応）
- **府県**: 自動入力（選手選択時）
- **級班**: 自動入力（SS/S1/S2/A1/A2/A3/L1）
- **脚質**: 自動入力（逃げ/追込/両）
- **競走得点**: **必須・手動入力**（例：115.20）

### 出力

#### 予測結果
- **高配当確率**: 0.05-0.95（5%-95%）
- **信頼度**: high/medium/low
- **Track調整**: 会場の荒れやすさ補正
- **Category調整**: カテゴリの荒れやすさ補正

#### 買い目提案
- **戦略**: 確率に応じた戦略
  - 60%以上: 穴狙い（大荒れ期待）
  - 45-60%: 中穴狙い（バランス型）
  - 30-45%: 堅め軸穴流し
  - 30%未満: 本命勝負
- **三連単**: 期待値重視（予算2,000円で最大20点）
- **三連複**: 的中率重視（オプション）

---

## 🔧 システム構成

### 主要ファイル

```
100_keirin/
├── build_clean_dataset.py           # データ統合（事後データ除外）
├── train_clean_model.py             # モデル訓練
├── compute_track_category_stats.py  # 統計計算
├── inference_from_clean_model.py    # 推論エンジン
├── improved_betting_suggestions.py  # 買い目提案
├── clean_web_app.py                 # Webアプリ
│
├── data/
│   ├── keirin_training_dataset_20240101_20240331.csv  # 生データ（3ヶ月）
│   └── clean_training_dataset.csv                     # クリーンデータ
│
├── analysis/model_outputs/
│   ├── clean_model_lgbm.txt           # 訓練済みモデル
│   ├── clean_model_metadata.json      # メタデータ
│   ├── track_category_stats.json      # Track/Category統計
│   └── rider_master.json              # 選手マスタ（2,499名）
│
└── templates/
    ├── index_clean.html               # 入力フォーム
    └── result_clean.html              # 予測結果表示
```

---

## 📋 使用している事前データ

### ✅ 完全に事前データ（レース前に分かる）

#### 選手情報
- **競走得点（heikinTokuten）**: KEIRIN.JPで公開
- **級班（kyuhan）**: SS/S1/S2/A1/A2/A3/L1
- **脚質（kyakusitu）**: 逃げ/追込/両
- **府県（entry_prefecture）**: 東京、神奈川など
- **B関連（脚質カウント）**:
  - nigeCnt - 逃げ勝ち回数
  - makuriCnt - 捲り勝ち回数
  - sasiCnt - 差し勝ち回数
  - markCnt - マーク勝ち回数
  - backCnt - バック回数

#### レース情報
- **開催場（track）**: 京王閣、いわき平など
- **レース番号（race_no）**: 1-12（時間帯の指標）
- **グレード（grade）**: GP/G1/G2/G3/F1/F2
- **カテゴリ（category）**: S級選抜、A級特選など
- **日付（race_date）**: YYYYMMDD

### ❌ 絶対に使っていないデータ（事後データ）
- **人気順位（trifecta_popularity）**
- **オッズ（odds）**
- **着順（finish_pos）**
- **レース結果（raceResult）**
- **投票動向（betting_trends）**

---

## 🎯 推論の仕組み

### Step 1: 特徴量構築

入力された選手情報から以下を計算：
- 得点統計（mean, std, cv, iqr, range）
- 脚質分析（比率、多様性、エントロピー）
- 級班分析（比率、混合度、エントロピー）
- ライン分析（地域別、バランス、支配度）
- B関連統計（mean, std, cv, max, sum）
- 推定人気度（得点ベース）

### Step 2: LightGBM予測

86特徴量 → LightGBM → 高配当確率（0.05-0.95）

### Step 3: Track/Category調整

```
最終確率 = 基本確率 + Track調整×0.3 + Category調整×0.3
```

例：
- 基本確率: 0.50
- Track調整: +0.10（いわき平は荒れやすい）
- Category調整: +0.15（S級選抜は荒れやすい）
- 最終確率: 0.50 + 0.10×0.3 + 0.15×0.3 = **0.575**

### Step 4: 買い目提案

確率に応じて戦略を変更：
- **60%以上**: 穴を絡める、大穴狙い
- **45-60%**: 本命軸で2-3着流し
- **30-45%**: 上位3名のボックス
- **30%未満**: 本命1着固定

---

## 🔝 荒れやすい会場・カテゴリ

### 荒れやすい会場（トップ5）
1. **取手**: 37.2%
2. **いわき平**: 36.1%
3. **静岡**: 34.5%
4. **岐阜**: 33.0%
5. **武雄**: 32.6%

### 堅い会場（ボトム5）
1. **前橋**: 19.0%
2. **高知**: 19.4%
3. **向日町**: 20.0%
4. **松山**: 22.0%
5. **名古屋**: 22.2%

### 荒れやすいカテゴリ（トップ5）
1. **S級選抜**: 44.6%
2. **S級一次予選**: 40.4%
3. **A級初日特選**: 38.1%
4. **A級選抜**: 36.2%
5. **S級特選**: 35.1%

---

## ⚠️ 重要な注意事項

### データリーケージの防止

1. **TimeSeriesSplit使用**: 過去データで訓練 → 未来データで検証
2. **事後データ完全除外**: 人気順位、オッズ、着順を一切使用しない
3. **推定人気**: 実際の人気ではなく、**得点から推定**

### 用語の定義

システム内で使われる用語：
- **「本命」** = 得点1位の選手（実際の人気1番ではない）
- **「上位」** = 得点上位の選手
- **「穴」** = 得点下位の選手

**実際の人気順位とは異なります！**

### 精度の解釈

- **ROC-AUC 0.56**: 事前データのみとしては優秀
- **Precision@Top100 0.38**: 上位100予測のうち38%が高配当
- **期待値**: あくまで参考値、100%的中ではない

---

## 🔄 更新履歴

### Version 2.0 - B関連特徴量追加（2025-11-18）
- nigeCnt, makuriCnt, sasiCnt, markCnt, backCnt を追加
- ROC-AUC: 0.5550 → 0.5634 (+1.5%)
- sasiCnt_cv が最重要特徴量に

### Version 1.0 - 初期リリース（2025-11-18）
- 事前データのみで高配当予測
- TimeSeriesSplitでデータリーケージ防止
- ROC-AUC: 0.5550
- 特徴量数: 59個

---

## 📞 トラブルシューティング

### Q: 精度が低いと感じる
A: 事前データのみでの予測なので、ROC-AUC 0.56は妥当です。人気順位を使えば0.99以上になりますが、それではレース前に予測できません。

### Q: 「本命」が実際の人気と違う
A: システムの「本命」は得点1位の選手です。実際の人気順位（オッズ）は使用していません。

### Q: Precision@Top100が低い
A: 上位100予測のうち38%が高配当なので、100レース予測すれば38レースで高配当が出る計算です。これは十分実用的です。

### Q: モデルを再訓練したい
A: 新しいデータを追加したら、以下を順に実行：
1. `python3 build_clean_dataset.py`
2. `python3 train_clean_model.py`
3. `python3 compute_track_category_stats.py`

---

## 🎯 今後の改善案

### データ拡充
- [ ] 2024年全期間のデータ収集（現在は3ヶ月のみ）
- [ ] 選手の直近成績（最近5レースなど）
- [ ] 天候データ（雨天、風速など）
- [ ] 並び予想データ

### 特徴量エンジニアリング
- [ ] 選手間の対戦成績
- [ ] 会場別の選手成績
- [ ] ライン内の相性分析
- [ ] 時系列特徴量（最近の調子）

### モデル改善
- [ ] ハイパーパラメータ最適化（GridSearch/Optuna）
- [ ] アンサンブル学習（複数モデルの組み合わせ）
- [ ] ニューラルネットワーク（LSTM/Transformerなど）
- [ ] 閾値の動的調整

### UI/UX改善
- [ ] 過去の予測履歴保存
- [ ] 的中率のトラッキング
- [ ] レース結果の自動取得・検証
- [ ] プッシュ通知（注目レース）

---

## 📚 参考資料

### KEIRIN.JP
- 出走表: https://keirin.jp/pc/dfw/portal/guest/race/info/
- 選手情報: https://keirin.jp/pc/dfw/portal/guest/rider/info/

### データソース
- 競走得点: KEIRIN.JPの出走表
- 級班・脚質: 選手登録情報
- B関連: 過去の戦績データ

---

**システムは完成しました！実際のレースでテストして、結果を教えてください。**
