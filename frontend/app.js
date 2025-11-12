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

// 初期化
document.addEventListener('DOMContentLoaded', async () => {
    await loadReferenceData();
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
                <input type="text" id="rider${i}_name" name="rider${i}_name" placeholder="例: 山田 太郎" required>
                <small class="region-note">地域は自動的に取得されます</small>
            </div>
        `;
        ridersContainer.appendChild(riderDiv);
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
    // 総組み合わせ数
    document.getElementById('totalCombinations').textContent = result.total_combinations;

    // 高配当リスト
    const highPayoutList = document.getElementById('highPayoutList');
    highPayoutList.innerHTML = '';
    result.high_payout_combinations.forEach(combo => {
        const item = createCombinationItem(combo);
        highPayoutList.appendChild(item);
    });

    // 低配当リスト
    const lowPayoutList = document.getElementById('lowPayoutList');
    lowPayoutList.innerHTML = '';
    result.low_payout_combinations.forEach(combo => {
        const item = createCombinationItem(combo);
        lowPayoutList.appendChild(item);
    });

    // 結果を表示
    showResult();

    // 結果までスクロール
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 組み合わせアイテムを作成
function createCombinationItem(combo) {
    const div = document.createElement('div');
    div.className = 'combination-item';

    const probabilityPercent = (combo.probability * 100).toFixed(1);
    const labelClass = combo.prediction === 1 ? 'high' : 'low';

    div.innerHTML = `
        <div class="combination-header">
            <span class="rank">#${combo.rank}</span>
            <span class="combination">${combo.combination}</span>
            <span class="probability">${probabilityPercent}%</span>
            <span class="label ${labelClass}">${combo.prediction_label}</span>
        </div>
        <div class="combination-details">
            <div class="riders-info">
                ${combo.riders.map((rider, idx) => `
                    <span class="rider">
                        <strong>${rider}</strong>
                        <small>(${combo.regions[idx]})</small>
                    </span>
                `).join(' → ')}
            </div>
        </div>
    `;

    return div;
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
