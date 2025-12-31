// Training data: y = 2x + 1
const X = [1, 2, 3, 4, 5];
const Y = [3, 5, 7, 9, 11];

// Model parameters
let weight = 0.0;
let bias = 0.0;
const learningRate = 0.01;
let epoch = 0;
let isTraining = false;
let trainingInterval = null;

// Chart data
const lossHistory = [];
const weightHistory = [];
const biasHistory = [];

// Chart.js global styling for dark theme
Chart.defaults.color = '#666';
Chart.defaults.borderColor = '#222';

// Initialize Charts
const lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Loss',
            data: [],
            borderColor: '#ffffff',
            backgroundColor: 'rgba(255,255,255,0.05)',
            fill: true,
            tension: 0.4,
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: '#1a1a1a' },
                ticks: { color: '#555' }
            },
            x: {
                grid: { color: '#1a1a1a' },
                ticks: { color: '#555' }
            }
        },
        plugins: { legend: { display: false } }
    }
});

const modelChart = new Chart(document.getElementById('modelChart'), {
    type: 'scatter',
    data: {
        datasets: [
            {
                label: 'Training Data',
                data: X.map((x, i) => ({ x, y: Y[i] })),
                backgroundColor: '#ffffff',
                pointRadius: 6,
                pointHoverRadius: 8
            },
            {
                label: 'Model Prediction',
                data: [],
                borderColor: '#666',
                backgroundColor: 'transparent',
                type: 'line',
                pointRadius: 0,
                borderWidth: 2
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
            x: {
                min: 0,
                max: 6,
                grid: { color: '#1a1a1a' },
                ticks: { color: '#555' }
            },
            y: {
                min: 0,
                max: 15,
                grid: { color: '#1a1a1a' },
                ticks: { color: '#555' }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#666' }
            }
        }
    }
});

function trainStep() {
    // Forward pass
    const predictions = X.map(x => weight * x + bias);

    // Calculate loss (MSE)
    const errors = predictions.map((p, i) => p - Y[i]);
    const loss = errors.reduce((sum, e) => sum + e * e, 0) / errors.length;

    // Backward pass (gradients)
    const weightGrad = errors.reduce((sum, e, i) => sum + 2 * e * X[i], 0) / errors.length;
    const biasGrad = errors.reduce((sum, e) => sum + 2 * e, 0) / errors.length;

    // Update weights
    weight -= learningRate * weightGrad;
    bias -= learningRate * biasGrad;

    epoch++;

    // Store history
    if (epoch % 10 === 0 || epoch <= 10) {
        lossHistory.push(loss);
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update('none');
    }

    // Update model line
    modelChart.data.datasets[1].data = [{ x: 0, y: bias }, { x: 6, y: weight * 6 + bias }];
    modelChart.update('none');

    // Update stats
    document.getElementById('epochVal').textContent = epoch;
    document.getElementById('weightVal').textContent = weight.toFixed(3);
    document.getElementById('biasVal').textContent = bias.toFixed(3);
    document.getElementById('lossVal').textContent = loss.toFixed(4);
    document.getElementById('predictVal').textContent = (weight * 10 + bias).toFixed(2);

    const accuracy = Math.max(0, 100 - loss * 10).toFixed(1);
    document.getElementById('accuracyVal').textContent = accuracy + '%';

    // Log every 100 epochs
    if (epoch % 100 === 0) {
        addTrainingLog(`Epoch ${epoch} | Loss: ${loss.toFixed(4)} | w: ${weight.toFixed(3)} | b: ${bias.toFixed(3)}`);
    }

    // Stop if converged
    if (loss < 0.0001 || epoch >= 1000) {
        stopTraining();
        addTrainingLog(`Training complete. Learned: y = ${weight.toFixed(2)}x + ${bias.toFixed(2)}`);
    }
}

function startTraining() {
    if (isTraining) return;
    isTraining = true;
    document.getElementById('startBtn').disabled = true;
    addTrainingLog('Training initiated...');
    trainingInterval = setInterval(trainStep, 50);
}

function stopTraining() {
    isTraining = false;
    document.getElementById('startBtn').disabled = false;
    if (trainingInterval) clearInterval(trainingInterval);
}

function stepTraining() {
    if (!isTraining) trainStep();
}

function resetTraining() {
    stopTraining();
    weight = 0; bias = 0; epoch = 0;
    lossHistory.length = 0;
    lossChart.data.labels = [];
    lossChart.data.datasets[0].data = [];
    lossChart.update();
    modelChart.data.datasets[1].data = [];
    modelChart.update();
    document.getElementById('epochVal').textContent = '0';
    document.getElementById('weightVal').textContent = '0.00';
    document.getElementById('biasVal').textContent = '0.00';
    document.getElementById('lossVal').textContent = '0.00';
    document.getElementById('predictVal').textContent = '-';
    document.getElementById('accuracyVal').textContent = '0%';
    document.getElementById('trainingLog').innerHTML = '';
    addTrainingLog('System reset. Ready.');
}

function addTrainingLog(msg) {
    const log = document.getElementById('trainingLog');
    log.innerHTML += `<div class="epoch-line">${msg}</div>`;
    log.scrollTop = log.scrollHeight;
}

// Question logging
const questions = JSON.parse(localStorage.getItem('aiQuestions') || '[]');
renderQuestions();

function logQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();
    if (!question) return;

    const entry = { time: new Date().toLocaleString(), question };
    questions.unshift(entry);
    localStorage.setItem('aiQuestions', JSON.stringify(questions));
    renderQuestions();
    input.value = '';
}

function renderQuestions() {
    const log = document.getElementById('questionLog');
    log.innerHTML = questions.map(q =>
        `<div class="log-entry"><span class="log-time">${q.time}</span>${q.question}</div>`
    ).join('');
}

document.getElementById('questionInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') logQuestion();
});

addTrainingLog('System initialized. Click "Start Training" to begin.');

