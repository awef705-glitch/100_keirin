# 🚀 クイックスタート - Cloudflare Tunnel

**完全無料・デプロイ不要でiPhoneから競輪予測アプリを使う方法**

---

## 📱 3ステップで起動

### 1️⃣ cloudflaredをインストール（初回のみ）

**WSL/Linux**:
```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

**Windows**:
[こちらからダウンロード](https://github.com/cloudflare/cloudflared/releases)

### 2️⃣ アプリを起動

**WSL/Linux**:
```bash
./start_tunnel.sh
```

**Windows**:
```cmd
start_tunnel.bat
```

### 3️⃣ iPhoneでアクセス

表示されたURL（例: `https://abc-def-ghi.trycloudflare.com`）をiPhoneのSafariで開く

---

## ✅ これだけでOK

- ✅ デプロイ不要（PCで動かすだけ）
- ✅ 完全無料・無制限
- ✅ デバッグが簡単（ローカル実行）
- ✅ 無料枠の心配なし

---

## 🛑 停止方法

ターミナルで `Ctrl+C` を押す（サーバーも自動停止）

---

## ⚙️ 詳細設定

固定URLや自動起動など、詳しくは `CLOUDFLARE_TUNNEL_SETUP.md` を参照してください。

---

## ⚠️ 注意

- PCが起動している間のみ利用可能
- インターネット接続が必要
- URLは起動ごとに変わる（固定URL版は `CLOUDFLARE_TUNNEL_SETUP.md` 参照）
