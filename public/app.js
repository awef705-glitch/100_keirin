// 選手カウンター
let riderCount = 0;

// 初期化
document.addEventListener('DOMContentLoaded', () => {
  // 今日の日付をデフォルトに設定
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('race_date').value = today;

  // 初期選手を9名追加
  for (let i = 0; i < 9; i++) {
    addRider();
  }

  // フォーム送信イベント
  document.getElementById('predictionForm').addEventListener('submit', handleSubmit);
});

// 選手を追加
function addRider() {
  riderCount++;
  const container = document.getElementById('ridersContainer');

  const riderCard = document.createElement('div');
  riderCard.className = 'rider-card';
  riderCard.id = `rider-${riderCount}`;

  riderCard.innerHTML = `
    <div class="rider-header">
      <span class="rider-number">${riderCount}番車</span>
      ${riderCount > 1 ? `<button type="button" class="btn-remove" onclick="removeRider(${riderCount})">削除</button>` : ''}
    </div>
    <div class="form-grid">
      <div class="form-group full">
        <label>選手名</label>
        <input type="text" name="rider_${riderCount}_name" placeholder="例: 山田太郎" required>
      </div>
      <div class="form-group">
        <label>競走得点</label>
        <input type="number" name="rider_${riderCount}_score" step="0.01" placeholder="例: 95.50" required>
      </div>
      <div class="form-group">
        <label>階級</label>
        <select name="rider_${riderCount}_grade" required>
          <option value="">選択</option>
          <option value="SS">SS級</option>
          <option value="S1">S1級</option>
          <option value="S2">S2級</option>
          <option value="A1">A1級</option>
          <option value="A2">A2級</option>
          <option value="A3">A3級</option>
          <option value="L1">L1級</option>
        </select>
      </div>
      <div class="form-group">
        <label>脚質</label>
        <select name="rider_${riderCount}_style" required>
          <option value="">選択</option>
          <option value="nige">逃げ</option>
          <option value="tsui">追込</option>
          <option value="ryo">自在</option>
        </select>
      </div>
      <div class="form-group full">
        <label>都道府県</label>
        <input type="text" name="rider_${riderCount}_prefecture" placeholder="例: 東京" required>
      </div>
    </div>
  `;

  container.appendChild(riderCard);
}

// 選手を削除
function removeRider(id) {
  const riderCard = document.getElementById(`rider-${id}`);
  if (riderCard) {
    riderCard.remove();
  }
}

// フォーム送信処理
async function handleSubmit(e) {
  e.preventDefault();

  const submitBtn = document.getElementById('submitBtn');
  const submitText = document.getElementById('submitText');
  const submitLoader = document.getElementById('submitLoader');
  const resultArea = document.getElementById('resultArea');

  // ローディング状態
  submitBtn.disabled = true;
  submitText.style.display = 'none';
  submitLoader.style.display = 'block';
  resultArea.style.display = 'none';

  try {
    // フォームデータを収集
    const formData = new FormData(e.target);
    const data = collectFormData(formData);

    // API呼び出し
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    // 結果を表示
    displayResult(result);

  } catch (error) {
    console.error('予測エラー:', error);
    alert('予測に失敗しました。入力内容を確認してください。\n\nエラー: ' + error.message);
  } finally {
    // ローディング解除
    submitBtn.disabled = false;
    submitText.style.display = 'block';
    submitLoader.style.display = 'none';
  }
}

// フォームデータを収集
function collectFormData(formData) {
  // レース基本情報
  const raceDate = formData.get('race_date').replace(/-/g, ''); // YYYYMMDD形式に変換
  const data = {
    race_date: parseInt(raceDate),
    race_no: parseInt(formData.get('race_no')),
    track: formData.get('track'),
    grade: formData.get('grade') || '',
    category: formData.get('category'),
    riders: []
  };

  // 選手情報を収集
  const riderCards = document.querySelectorAll('.rider-card');
  riderCards.forEach((card, index) => {
    const riderNum = card.id.split('-')[1];
    const rider = {
      name: formData.get(`rider_${riderNum}_name`),
      score: parseFloat(formData.get(`rider_${riderNum}_score`)),
      grade: formData.get(`rider_${riderNum}_grade`),
      style: formData.get(`rider_${riderNum}_style`),
      prefecture: formData.get(`rider_${riderNum}_prefecture`)
    };
    data.riders.push(rider);
  });

  return data;
}

// 結果を表示
function displayResult(result) {
  const resultArea = document.getElementById('resultArea');
  const resultContent = document.getElementById('resultContent');

  // スコアと信頼度
  const score = Math.round(result.roughness_score);
  const probability = (result.high_payout_probability * 100).toFixed(1);
  const confidence = getConfidenceLevel(result.high_payout_probability);

  let html = `
    <div class="result-score">
      <div class="score-label">荒れ度スコア</div>
      <div class="score-value">${score}</div>
      <div class="score-label">高配当確率: ${probability}%</div>
      <div class="confidence ${confidence.class}">${confidence.label}</div>
    </div>
  `;

  // 理由
  if (result.reasons && result.reasons.length > 0) {
    html += `
      <h3 style="margin-top: 24px; margin-bottom: 12px; font-size: 1rem;">📊 荒れる理由</h3>
      <ul class="reasons-list">
        ${result.reasons.map(reason => `<li>${reason}</li>`).join('')}
      </ul>
    `;
  }

  // 買い目提案
  if (result.suggestions && result.suggestions.length > 0) {
    html += `
      <div class="suggestions">
        <h3 style="margin-bottom: 12px; font-size: 1rem;">💡 買い目提案</h3>
        ${result.suggestions.map(suggestion => `
          <div class="suggestion-item">${suggestion}</div>
        `).join('')}
      </div>
    `;
  }

  resultContent.innerHTML = html;
  resultArea.style.display = 'block';

  // 結果エリアまでスクロール
  resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 信頼度レベルを取得
function getConfidenceLevel(probability) {
  if (probability >= 0.7) {
    return { class: 'high', label: '信頼度: 高' };
  } else if (probability >= 0.5) {
    return { class: 'medium', label: '信頼度: 中' };
  } else {
    return { class: 'low', label: '信頼度: 低' };
  }
}

// PWA: Service Worker登録
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js')
      .then(registration => console.log('SW registered'))
      .catch(err => console.log('SW registration failed'));
  });
}
