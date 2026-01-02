// 選手マスターデータ
let riderMaster = [];
let selectedRiders = [];

// 初期化
document.addEventListener('DOMContentLoaded', async () => {
  // 選手データを読み込み
  try {
    const res = await fetch('riders.json');
    riderMaster = await res.json();
    console.log(`Loaded ${riderMaster.length} riders`);
  } catch (e) {
    console.error('Failed to load riders:', e);
    alert('選手データの読み込みに失敗しました');
  }

  // 9枠分の入力欄を作成
  const container = document.getElementById('ridersContainer');
  for (let i = 1; i <= 9; i++) {
    container.appendChild(createRiderInput(i));
  }

  // フォーム送信イベント
  document.getElementById('predictionForm').addEventListener('submit', handleSubmit);
});

// 選手入力欄を作成
function createRiderInput(num) {
  const div = document.createElement('div');
  div.className = 'rider-input-row';
  div.innerHTML = `
    <span class="waku-number waku-${num}">${num}</span>
    <div class="autocomplete-wrapper">
      <input type="text"
             id="rider-${num}"
             class="rider-search"
             placeholder="選手名を入力..."
             autocomplete="off"
             data-waku="${num}">
      <div id="dropdown-${num}" class="autocomplete-dropdown"></div>
    </div>
    <div id="rider-info-${num}" class="rider-info"></div>
  `;

  // 入力イベント
  setTimeout(() => {
    const input = document.getElementById(`rider-${num}`);
    const dropdown = document.getElementById(`dropdown-${num}`);

    input.addEventListener('input', () => {
      const query = input.value.trim();
      if (query.length < 1) {
        dropdown.style.display = 'none';
        return;
      }
      showDropdown(query, num);
    });

    input.addEventListener('focus', () => {
      if (input.value.trim().length >= 1) {
        showDropdown(input.value.trim(), num);
      }
    });

    input.addEventListener('blur', () => {
      setTimeout(() => dropdown.style.display = 'none', 200);
    });
  }, 0);

  return div;
}

// ドロップダウンを表示
function showDropdown(query, waku) {
  const dropdown = document.getElementById(`dropdown-${waku}`);
  const q = query.toLowerCase();

  // 検索（名前、都道府県で部分一致）
  const matches = riderMaster
    .filter(r => r.name.toLowerCase().includes(q) || r.pref.includes(query))
    .slice(0, 10);

  if (matches.length === 0) {
    dropdown.innerHTML = '<div class="dropdown-item no-match">該当なし</div>';
  } else {
    dropdown.innerHTML = matches.map(r => `
      <div class="dropdown-item" onclick="selectRider(${waku}, '${r.name.replace(/'/g, "\\'")}')">
        <span class="rider-name">${r.name}</span>
        <span class="rider-meta">${r.pref} ${r.grade} ${r.score.toFixed(1)}点</span>
      </div>
    `).join('');
  }
  dropdown.style.display = 'block';
}

// 選手を選択
function selectRider(waku, name) {
  const rider = riderMaster.find(r => r.name === name);
  if (!rider) return;

  const input = document.getElementById(`rider-${waku}`);
  const info = document.getElementById(`rider-info-${waku}`);
  const dropdown = document.getElementById(`dropdown-${waku}`);

  input.value = rider.name;
  dropdown.style.display = 'none';

  // 選手情報を表示
  info.innerHTML = `
    <span class="tag grade-${rider.grade}">${rider.grade}</span>
    <span class="tag">${rider.pref}</span>
    <span class="tag style-${rider.style}">${rider.style}</span>
    <span class="tag score">${rider.score.toFixed(1)}</span>
  `;
  info.dataset.rider = JSON.stringify(rider);
}

// フォーム送信
function handleSubmit(e) {
  e.preventDefault();

  const submitBtn = document.getElementById('submitBtn');
  const submitText = document.getElementById('submitText');
  const submitLoader = document.getElementById('submitLoader');
  const resultArea = document.getElementById('resultArea');

  // 選手データを収集
  const riders = [];
  for (let i = 1; i <= 9; i++) {
    const info = document.getElementById(`rider-info-${i}`);
    if (info.dataset.rider) {
      riders.push(JSON.parse(info.dataset.rider));
    }
  }

  if (riders.length < 7) {
    alert('最低7名の選手を選択してください');
    return;
  }

  submitBtn.disabled = true;
  submitText.style.display = 'none';
  submitLoader.style.display = 'block';
  resultArea.style.display = 'none';

  setTimeout(() => {
    try {
      const data = {
        track: document.getElementById('track').value,
        grade: document.getElementById('grade').value,
        riders: riders
      };
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

// 予測ロジック
function predict(data) {
  const riders = data.riders;
  const scores = riders.map(r => r.score);
  const styles = riders.map(r => r.style);
  const grades = riders.map(r => r.grade);
  const prefs = riders.map(r => r.pref);

  // 統計計算
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / scores.length;
  const std = Math.sqrt(variance);
  const cv = (std / mean) * 100;
  const range = Math.max(...scores) - Math.min(...scores);

  // 多様性
  const styleDiversity = new Set(styles).size;
  const gradeDiversity = new Set(grades).size;
  const prefDiversity = new Set(prefs).size;

  // スコア計算
  let score = 30;
  const reasons = [];

  // 競走得点のばらつき
  if (cv > 8) {
    score += 25;
    reasons.push(`競走得点のばらつきが大きい（CV: ${cv.toFixed(1)}%）`);
  } else if (cv > 5) {
    score += 15;
    reasons.push(`競走得点にやや差がある（CV: ${cv.toFixed(1)}%）`);
  } else if (cv < 3) {
    score -= 10;
    reasons.push(`競走得点が接近している（本命決着傾向）`);
  }

  // 得点レンジ
  if (range > 15) {
    score += 15;
    reasons.push(`実力差が大きい（最大差: ${range.toFixed(1)}点）`);
  } else if (range > 10) {
    score += 8;
  }

  // 脚質の多様性
  if (styleDiversity >= 3) {
    score += 10;
    reasons.push(`脚質がバラバラ（展開が読みにくい）`);
  } else if (styleDiversity === 1) {
    score -= 5;
  }

  // グレード
  const gradeVal = data.grade;
  if (gradeVal === 'F2' || gradeVal === '一般') {
    score += 12;
    reasons.push(`${gradeVal}グレードは荒れやすい傾向`);
  } else if (gradeVal === 'GP' || gradeVal === 'G1') {
    score -= 8;
    reasons.push(`${gradeVal}は実力通りになりやすい`);
  }

  // 都道府県
  if (prefDiversity >= 7) {
    score += 8;
    reasons.push(`出身地がバラバラ（ライン読みにくい）`);
  } else if (prefDiversity <= 3) {
    score -= 5;
    reasons.push(`同郷選手が多い（ライン明確）`);
  }

  // 階級混在
  if (gradeDiversity >= 4) {
    score += 8;
    reasons.push(`階級が混在している`);
  }

  score = Math.max(0, Math.min(100, score));

  // 買い目提案
  const suggestions = [];
  if (score >= 70) {
    suggestions.push('三連単ボックスで穴狙い推奨');
    suggestions.push('2-3着に中位選手を入れた買い目');
  } else if (score >= 50) {
    suggestions.push('本命軸からのフォーメーション');
    suggestions.push('ワイドで押さえも検討');
  } else {
    suggestions.push('堅い決着の可能性が高い');
    suggestions.push('上位人気の組み合わせを中心に');
  }

  return { roughness_score: score, high_payout_probability: score / 100, reasons, suggestions };
}

// 結果表示
function displayResult(result) {
  const resultArea = document.getElementById('resultArea');
  const resultContent = document.getElementById('resultContent');
  const score = Math.round(result.roughness_score);
  const confidence = score >= 70 ? { class: 'high', label: '高' } :
                     score >= 50 ? { class: 'medium', label: '中' } :
                     { class: 'low', label: '低' };

  let html = `
    <div class="result-score">
      <div class="score-label">荒れ度スコア</div>
      <div class="score-value">${score}</div>
      <div class="confidence ${confidence.class}">信頼度: ${confidence.label}</div>
    </div>
  `;

  if (result.reasons.length > 0) {
    html += `<h3 style="margin-top: 24px; margin-bottom: 12px; font-size: 1rem;">分析結果</h3>
      <ul class="reasons-list">${result.reasons.map(r => `<li>${r}</li>`).join('')}</ul>`;
  }

  if (result.suggestions.length > 0) {
    html += `<div class="suggestions"><h3 style="margin-bottom: 12px; font-size: 1rem;">買い目提案</h3>
      ${result.suggestions.map(s => `<div class="suggestion-item">${s}</div>`).join('')}</div>`;
  }

  resultContent.innerHTML = html;
  resultArea.style.display = 'block';
  resultArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
