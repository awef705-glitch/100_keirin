// API設定
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:5000'
    : '';

// DOM要素
const raceForm = document.getElementById('raceForm');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const submitBtn = document.getElementById('submitBtn');
const ridersContainer = document.getElementById('ridersContainer');
const riderCountSelect = document.getElementById('rider_count');

// 選手名リスト
let riderNamesList = [];

// 初期化
document.addEventListener('DOMContentLoaded', async () => {
    await loadReferenceData();
    await loadRiderNames();
    setupFormHandlers();
    generateRiderInputs(9); // デフォルト9人
});

// リファレンスデータの読み込み
async function loadReferenceData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/reference-data`);
        const data = await response.json();

        // 場名の選択肢を設定
        const trackSelect = document.getElementById('track');
        data.tracks.forEach(track => {
            const option = document.createElement('option');
            option.value = track;
            option.textContent = track;
            trackSelect.appendChild(option);
        });

        // グレードの選択肢を設定
        const gradeSelect = document.getElementById('grade');
        data.grades.forEach(grade => {
            const option = document.createElement('option');
            option.value = grade;
            option.textContent = grade;
            gradeSelect.appendChild(option);
        });

        // カテゴリーの選択肢を設定
        const categorySelect = document.getElementById('category');
        data.categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            categorySelect.appendChild(option);
        });

    } catch (error) {
        console.error('リファレンスデータの読み込みに失敗しました:', error);
        alert('データの読み込みに失敗しました。サーバーが起動しているか確認してください。');
    }
}

// 選手名リストの読み込み
async function loadRiderNames() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/rider-names`);
        riderNamesList = await response.json();
        console.log(`選手名リスト読み込み完了: ${riderNamesList.length}人`);
    } catch (error) {
        console.error('選手名リストの読み込みに失敗しました:', error);
    }
}

// 選手入力フォームを生成
function generateRiderInputs(count) {
    ridersContainer.innerHTML = '';

    for (let i = 1; i <= count; i++) {
        const riderDiv = document.createElement('div');
        riderDiv.className = 'rider-input';
        riderDiv.innerHTML = `
            <h4>車番 ${i}</h4>
            <div class="form-group">
                <label for="rider${i}_name">選手名 *</label>
                <input type="text"
                       id="rider${i}_name"
                       name="rider${i}_name"
                       list="rider-names-list"
                       placeholder="例: 山田 太郎"
                       autocomplete="off"
                       required>
                <small class="region-note">地域・脚質は自動取得されます</small>
            </div>
        `;
        ridersContainer.appendChild(riderDiv);
    }

    // datalist要素を一度だけ作成
    if (!document.getElementById('rider-names-list')) {
        const datalist = document.createElement('datalist');
        datalist.id = 'rider-names-list';
        riderNamesList.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            datalist.appendChild(option);
        });
        document.body.appendChild(datalist);
    }
}

// フォームハンドラーの設定
function setupFormHandlers() {
    // 出走人数変更時
    riderCountSelect.addEventListener('change', (e) => {
        generateRiderInputs(parseInt(e.target.value));
    });

    // フォーム送信
    raceForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await predictRace();
    });
}

// レース予測の実行
async function predictRace() {
    // フォームデータの取得
    const formData = new FormData(raceForm);

    // レース情報
    const raceInfo = {
        track: formData.get('track'),
        grade: formData.get('grade'),
        category: formData.get('category'),
        race_no: formData.get('race_no'),
        meeting_day: formData.get('meeting_day'),
        race_date: formData.get('race_date').replace(/-/g, '') // YYYY-MM-DD → YYYYMMDD
    };

    // 選手情報（地域はサーバー側で自動取得）
    const riderCount = parseInt(formData.get('rider_count'));
    const riders = [];
    for (let i = 1; i <= riderCount; i++) {
        riders.push({
            car_no: i,
            name: formData.get(`rider${i}_name`)
        });
    }

    const data = {
        ...raceInfo,
        riders: riders
    };

    // ローディング表示
    showLoading();
    hideResult();
    disableSubmit();

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (result.success) {
            displayResult(result);
        } else {
            alert('予測に失敗しました: ' + result.error);
        }

    } catch (error) {
        console.error('予測エラー:', error);
        alert('予測に失敗しました。ネットワーク接続を確認してください。');
    } finally {
        hideLoading();
        enableSubmit();
    }
}

// 結果の表示
function displayResult(result) {
    // レース荒れ度
    const probability = result.race_roughness_probability;
    const probabilityPercent = (probability * 100).toFixed(1);

    document.getElementById('roughnessProbability').textContent = probabilityPercent;
    document.getElementById('roughnessLevel').textContent = result.roughness_level;

    // 荒れ度バー
    const roughnessBar = document.getElementById('roughnessBar');
    roughnessBar.style.width = `${probabilityPercent}%`;

    // バーの色を確率に応じて変更
    if (probability >= 0.7) {
        roughnessBar.style.backgroundColor = '#f44336'; // 赤 - 超高配当
    } else if (probability >= 0.5) {
        roughnessBar.style.backgroundColor = '#ff9800'; // オレンジ - 高配当
    } else if (probability >= 0.3) {
        roughnessBar.style.backgroundColor = '#ffc107'; // 黄 - やや荒れる
    } else {
        roughnessBar.style.backgroundColor = '#4caf50'; // 緑 - 堅い
    }

    // パターン分析
    const patternAnalysis = document.getElementById('patternAnalysis');
    patternAnalysis.innerHTML = '';

    const patterns = result.pattern_analysis;
    const patternItems = [
        { icon: '🏃', label: '逃げ型選手', value: `${patterns.nige_count}人` },
        { icon: '⚡', label: '差し型選手', value: `${patterns.sashi_count}人` },
        { icon: '🌀', label: '捲り型選手', value: `${patterns.makuri_count}人` },
        { icon: '🌏', label: '主要地域ライン', value: patterns.major_regions.join(', ') || 'なし' },
        { icon: '🏠', label: 'ホーム選手', value: `${patterns.home_advantage_count}人` }
    ];

    patternItems.forEach(item => {
        const div = document.createElement('div');
        div.className = 'pattern-item';
        div.innerHTML = `
            <span class="pattern-icon">${item.icon}</span>
            <span class="pattern-label">${item.label}:</span>
            <strong class="pattern-value">${item.value}</strong>
        `;
        patternAnalysis.appendChild(div);
    });

    // 買い方提案
    const bettingSuggestions = document.getElementById('bettingSuggestions');
    bettingSuggestions.innerHTML = '';

    result.betting_suggestions.forEach((suggestion, idx) => {
        const div = document.createElement('div');
        div.className = 'betting-item';
        div.innerHTML = `
            <div class="betting-header">
                <span class="betting-rank">${idx + 1}</span>
                <span class="betting-type">${suggestion.ticket_type}</span>
            </div>
            <p class="betting-reason">${suggestion.reason}</p>
            ${suggestion.combinations && suggestion.combinations.length > 0 ? `
                <div class="betting-combinations">
                    <strong>推奨組み合わせ例:</strong>
                    ${suggestion.combinations.map(combo => `
                        <span class="combo-tag">${combo}</span>
                    `).join('')}
                </div>
            ` : ''}
        `;
        bettingSuggestions.appendChild(div);
    });

    // 結果を表示
    showResult();

    // 結果までスクロール
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ローディング表示/非表示
function showLoading() {
    loadingDiv.classList.remove('hidden');
}

function hideLoading() {
    loadingDiv.classList.add('hidden');
}

// 結果表示/非表示
function showResult() {
    resultDiv.classList.remove('hidden');
}

function hideResult() {
    resultDiv.classList.add('hidden');
}

// 送信ボタンの有効/無効化
function disableSubmit() {
    submitBtn.disabled = true;
    submitBtn.textContent = '予測中...';
}

function enableSubmit() {
    submitBtn.disabled = false;
    submitBtn.textContent = '全組み合わせを予測';
}

// PWAサービスワーカーの登録
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js')
        .then(registration => {
            console.log('Service Worker registered:', registration);
        })
        .catch(error => {
            console.log('Service Worker registration failed:', error);
        });
}
