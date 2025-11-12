// API設定
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:5000'
    : '';

// DOM要素
const raceForm = document.getElementById('raceForm');
const resultDiv = document.getElementById('result');
const loadingDiv = document.getElementById('loading');
const submitBtn = document.getElementById('submitBtn');

// 初期化
document.addEventListener('DOMContentLoaded', async () => {
    await loadReferenceData();
    setupFormHandlers();
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

// フォームハンドラーの設定
function setupFormHandlers() {
    raceForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        await predictRace();
    });
}

// レース予測の実行
async function predictRace() {
    // フォームデータの取得
    const formData = new FormData(raceForm);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    // 日付をYYYYMMDD形式に変換（YYYY-MM-DD → YYYYMMDD）
    if (data.race_date) {
        data.race_date = data.race_date.replace(/-/g, '');
    }

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
    const probability = result.probability;
    const probabilityPercent = (probability * 100).toFixed(1);

    // 確率の表示
    document.getElementById('probabilityValue').textContent = `${probabilityPercent}%`;
    document.getElementById('probabilityFill').style.width = `${probabilityPercent}%`;

    // 予測ラベルの表示
    const predictionLabel = document.getElementById('predictionLabel');
    predictionLabel.textContent = result.prediction_label;
    predictionLabel.className = 'prediction-label ' + (result.prediction === 1 ? 'high' : 'low');

    // 信頼度の表示
    const confidenceText = `信頼度: ${result.betting_strategy.confidence}`;
    document.getElementById('confidence').textContent = confidenceText;

    // 買い方の提案を表示
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '';

    result.betting_strategy.recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';

        const title = document.createElement('h4');
        title.textContent = rec.type;
        item.appendChild(title);

        const description = document.createElement('p');
        description.textContent = rec.description;
        item.appendChild(description);

        const numbers = document.createElement('div');
        numbers.className = 'recommendation-numbers';
        numbers.textContent = `推奨車番: ${rec.suggested_numbers.join('-')} (${rec.bet_type})`;
        item.appendChild(numbers);

        recommendationsDiv.appendChild(item);
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
    submitBtn.textContent = '予測する';
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
