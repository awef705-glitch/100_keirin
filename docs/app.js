// 選手カウンター
let riderCount = 0;

// 初期化
document.addEventListener('DOMContentLoaded', () => {
  const today = new Date().toISOString().split('T')[0];
  document.getElementById('race_date').value = today;
  for (let i = 0; i < 9; i++) {
    addRider();
  }
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
          <option value="逃">逃げ</option>
          <option value="追">追込</option>
          <option value="両">自在</option>
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
  if (riderCard) riderCard.remove();
}

// フォーム送信処理
function handleSubmit(e) {
  e.preventDefault();
  const submitBtn = document.getElementById('submitBtn');
  const submitText = document.getElementById('submitText');
  const submitLoader = document.getElementById('submitLoader');
  const resultArea = document.getElementById('resultArea');

  submitBtn.disabled = true;
  submitText.style.display = 'none';
  submitLoader.style.display = 'block';
  resultArea.style.display = 'none';

  setTimeout(() => {
    try {
      const formData = new FormData(e.target);
      const data = collectFormData(formData);
      const result = predict(data);
      displayResult(result);
    } catch (error) {
      alert('予測に失敗しました: ' + error.message);
    } finally {
      submitBtn.disabled = false;
      submitText.style.display = 'block';
      submitLoader.style.display = 'none';
    }
  }, 300);
}

// フォームデータを収集
function collectFormData(formData) {
  const raceDate = formData.get('race_date').replace(/-/g, '');
  const data = {
    race_date: parseInt(raceDate),
    race_no: parseInt(formData.get('race_no')),
    track: formData.get('track'),
    grade: formData.get('grade') || '',
    category: formData.get('category'),
    riders: []
  };
  const riderCards = document.querySelectorAll('.rider-card');
  riderCards.forEach((card) => {
    const riderNum = card.id.split('-')[1];
    data.riders.push({
      name: formData.get(`rider_${riderNum}_name`),
      score: parseFloat(formData.get(`rider_${riderNum}_score`)),
      grade: formData.get(`rider_${riderNum}_grade`),
      style: formData.get(`rider_${riderNum}_style`),
      prefecture: formData.get(`rider_${riderNum}_prefecture`)
    });
  });
  return data;
}

// 予測ロジック（クライアントサイド）
function predict(data) {
  const riders = data.riders;
  const scores = riders.map(r => r.score);
  const styles = riders.map(r => r.style);
  const grades = riders.map(r => r.grade);
  const prefs = riders.map(r => r.prefecture);

  // 統計計算
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / scores.length;
  const std = Math.sqrt(variance);
  const cv = (std / mean) * 100;
  const range = Math.max(...scores) - Math.min(...scores);

  // 脚質の多様性
  const styleSet = new Set(styles);
  const styleDiversity = styleSet.size;

  // 階級の多様性
  const gradeSet = new Set(grades);
  const gradeDiversity = gradeSet.size;

  // 都道府県の多様性
  const prefSet = new Set(prefs);
  const prefDiversity = prefSet.size;

  // スコア計算（0-100）
  let score = 30; // ベーススコア
  const reasons = [];

  // 競走得点のばらつき（最重要）
  if (cv > 8) {
    score += 25;
    reasons.push(`⚡ 競走得点のばらつきが大きい（CV: ${cv.toFixed(1)}%）`);
  } else if (cv > 5) {
    score += 15;
    reasons.push(`📊 競走得点にやや差がある（CV: ${cv.toFixed(1)}%）`);
  } else if (cv < 3) {
    score -= 10;
    reasons.push(`📉 競走得点が接近している（CV: ${cv.toFixed(1)}%）`);
  }

  // 得点レンジ
  if (range > 15) {
    score += 15;
    reasons.push(`📏 実力差が大きい（最大差: ${range.toFixed(1)}点）`);
  } else if (range > 10) {
    score += 8;
  }

  // 脚質の多様性
  if (styleDiversity >= 3) {
    score += 10;
    reasons.push(`🔄 脚質がバラバラ（${styleDiversity}種類）`);
  } else if (styleDiversity === 1) {
    score -= 5;
    reasons.push(`⚠️ 脚質が偏っている`);
  }

  // グレードによる調整
  const gradeVal = data.grade;
  if (gradeVal === 'F2' || gradeVal === '一般') {
    score += 12;
    reasons.push(`🎰 ${gradeVal}グレードは荒れやすい傾向`);
  } else if (gradeVal === 'GP' || gradeVal === 'G1') {
    score -= 8;
    reasons.push(`🏆 ${gradeVal}は実力通りになりやすい`);
  }

  // カテゴリによる調整
  const cat = data.category.toLowerCase();
  if (cat.includes('ガールズ') || cat.includes('girls')) {
    score += 10;
    reasons.push(`👩 ガールズケイリンは波乱傾向`);
  } else if (cat.includes('ヤング') || cat.includes('young')) {
    score += 8;
    reasons.push(`🌟 ヤング戦は予測困難`);
  } else if (cat.includes('特選')) {
    score -= 5;
  }

  // 都道府県の多様性
  if (prefDiversity >= 7) {
    score += 8;
    reasons.push(`🗾 出身地がバラバラ（ラインが読みにくい）`);
  } else if (prefDiversity <= 3) {
    score -= 5;
    reasons.push(`🤝 同郷選手が多い（ライン形成しやすい）`);
  }

  // 階級混在
  if (gradeDiversity >= 4) {
    score += 8;
    reasons.push(`🎭 階級が混在（予測困難）`);
  }

  // スコアを0-100に制限
  score = Math.max(0, Math.min(100, score));

  // 買い目提案
  const suggestions = [];
  if (score >= 70) {
    suggestions.push('💰 三連単ボックスで穴狙い推奨');
    suggestions.push('🎯 2-3着に中位選手を入れた買い目');
  } else if (score >= 50) {
    suggestions.push('📊 本命軸からのフォーメーション');
    suggestions.push('🔀 ワイドで押さえも検討');
  } else {
    suggestions.push('✅ 堅い決着の可能性が高い');
    suggestions.push('🎯 上位人気の組み合わせを中心に');
  }

  return {
    roughness_score: score,
    high_payout_probability: score / 100,
    reasons: reasons,
    suggestions: suggestions
  };
}

// 結果を表示
function displayResult(result) {
  const resultArea = document.getElementById('resultArea');
  const resultContent = document.getElementById('resultContent');
  const score = Math.round(result.roughness_score);
  const probability = (result.high_payout_probability * 100).toFixed(0);
  const confidence = getConfidenceLevel(result.high_payout_probability);

  let html = `
    <div class="result-score">
      <div class="score-label">荒れ度スコア</div>
      <div class="score-value">${score}</div>
      <div class="score-label">高配当確率: ${probability}%</div>
      <div class="confidence ${confidence.class}">${confidence.label}</div>
    </div>
  `;

  if (result.reasons && result.reasons.length > 0) {
    html += `
      <h3 style="margin-top: 24px; margin-bottom: 12px; font-size: 1rem;">📊 荒れる理由</h3>
      <ul class="reasons-list">
        ${result.reasons.map(reason => `<li>${reason}</li>`).join('')}
      </ul>
    `;
  }

  if (result.suggestions && result.suggestions.length > 0) {
    html += `
      <div class="suggestions">
        <h3 style="margin-bottom: 12px; font-size: 1rem;">💡 買い目提案</h3>
        ${result.suggestions.map(s => `<div class="suggestion-item">${s}</div>`).join('')}
      </div>
    `;
  }

  resultContent.innerHTML = html;
  resultArea.style.display = 'block';
  resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 信頼度レベルを取得
function getConfidenceLevel(probability) {
  if (probability >= 0.7) return { class: 'high', label: '信頼度: 高' };
  if (probability >= 0.5) return { class: 'medium', label: '信頼度: 中' };
  return { class: 'low', label: '信頼度: 低' };
}
