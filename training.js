// ============================================================
// EXAMPLE 1: Linear Regression (y = 2x + 1)
// ============================================================
const X = [1, 2, 3, 4, 5];
const Y = [3, 5, 7, 9, 11];

let weight = 0.0;
let bias = 0.0;
const learningRate = 0.01;
let epoch = 0;
let isTraining = false;
let trainingInterval = null;

const lossHistory = [];
const weightHistory = [];
const biasHistory = [];

// ============================================================
// EXAMPLE 2: Greeting Classifier (Neural Network)
// ============================================================
const greetingsData = {
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "good morning": "greeting",
    "good evening": "greeting",
    "bye": "farewell",
    "goodbye": "farewell",
    "see you": "farewell",
    "later": "farewell",
    "thanks": "gratitude",
    "thank you": "gratitude",
    "appreciate": "gratitude",
    "how are you": "question",
    "whats up": "question",
    "how do you do": "question"
};

const categories = ["greeting", "farewell", "gratitude", "question"];
const catToIdx = Object.fromEntries(categories.map((c, i) => [c, i]));
const idxToCat = Object.fromEntries(categories.map((c, i) => [i, c]));

// Build vocabulary
const allWords = new Set();
Object.keys(greetingsData).forEach(phrase => {
    phrase.split(" ").forEach(word => allWords.add(word));
});
const vocab = Array.from(allWords).sort();
const wordToIdx = Object.fromEntries(vocab.map((w, i) => [w, i]));
const vocabSize = vocab.length;

// Neural network weights (initialized randomly)
let W1, b1, W2, b2;
let greetingEpoch = 0;
let isGreetingTraining = false;
let greetingInterval = null;

function initGreetingNetwork() {
    const hiddenSize = 8;
    const outputSize = categories.length;

    W1 = Array(vocabSize).fill(0).map(() =>
        Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 0.5)
    );
    b1 = Array(hiddenSize).fill(0);
    W2 = Array(hiddenSize).fill(0).map(() =>
        Array(outputSize).fill(0).map(() => (Math.random() - 0.5) * 0.5)
    );
    b2 = Array(outputSize).fill(0);
    greetingEpoch = 0;
}

function textToVector(text) {
    const vector = Array(vocabSize).fill(0);
    text.toLowerCase().split(" ").forEach(word => {
        if (wordToIdx[word] !== undefined) {
            vector[wordToIdx[word]] = 1;
        }
    });
    return vector;
}

function relu(x) {
    return x.map(v => Math.max(0, v));
}

function softmax(x) {
    const max = Math.max(...x);
    const exp = x.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
}

function matMul(vec, mat) {
    return mat[0].map((_, j) =>
        vec.reduce((sum, v, i) => sum + v * mat[i][j], 0)
    );
}

function classifyGreeting(text) {
    if (!W1) return { category: "unknown", confidence: 0 };

    const x = textToVector(text);
    const z1 = matMul(x, W1).map((v, i) => v + b1[i]);
    const a1 = relu(z1);
    const z2 = matMul(a1, W2).map((v, i) => v + b2[i]);
    const a2 = softmax(z2);

    const maxIdx = a2.indexOf(Math.max(...a2));
    return {
        category: idxToCat[maxIdx],
        confidence: (a2[maxIdx] * 100).toFixed(1)
    };
}

// Response generator
const responseData = {
    "hello": "Hi there! How can I help you?",
    "hi": "Hello! Nice to meet you.",
    "hey": "Hey! What's going on?",
    "good morning": "Good morning! Hope you have a great day.",
    "good evening": "Good evening! How was your day?",
    "bye": "Goodbye! Take care.",
    "goodbye": "See you later! Have a nice day.",
    "thanks": "You're welcome!",
    "thank you": "My pleasure! Happy to help.",
    "how are you": "I'm doing great, thanks for asking!"
};

const defaultResponses = {
    "greeting": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a wonderful day!",
    "gratitude": "You're welcome! Glad I could help.",
    "question": "I'm doing well! Thanks for asking."
};

function getAIResponse(input) {
    const normalized = input.toLowerCase().trim();
    if (responseData[normalized]) {
        return responseData[normalized];
    }
    const { category } = classifyGreeting(normalized);
    return defaultResponses[category] || "I'm not sure how to respond to that.";
}

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

document.getElementById('greetingInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') testGreeting();
});

document.getElementById('chatInput').addEventListener('keypress', e => {
    if (e.key === 'Enter') sendChat();
});

// ============================================================
// Greeting Network Training
// ============================================================
initGreetingNetwork();

function trainGreetingStep() {
    const hiddenSize = 8;
    const outputSize = categories.length;
    const lr = 0.1;

    // Create training data
    const phrases = Object.keys(greetingsData);
    const labels = Object.values(greetingsData);

    // Forward pass for all samples
    let totalLoss = 0;
    let correct = 0;

    // Accumulate gradients
    let dW1 = Array(vocabSize).fill(0).map(() => Array(hiddenSize).fill(0));
    let db1 = Array(hiddenSize).fill(0);
    let dW2 = Array(hiddenSize).fill(0).map(() => Array(outputSize).fill(0));
    let db2 = Array(outputSize).fill(0);

    phrases.forEach((phrase, idx) => {
        const x = textToVector(phrase);
        const yTrue = catToIdx[labels[idx]];

        // Forward
        const z1 = matMul(x, W1).map((v, i) => v + b1[i]);
        const a1 = relu(z1);
        const z2 = matMul(a1, W2).map((v, i) => v + b2[i]);
        const a2 = softmax(z2);

        // Loss
        totalLoss -= Math.log(a2[yTrue] + 1e-8);
        if (a2.indexOf(Math.max(...a2)) === yTrue) correct++;

        // Backward
        const dz2 = a2.map((v, i) => v - (i === yTrue ? 1 : 0));

        for (let i = 0; i < hiddenSize; i++) {
            for (let j = 0; j < outputSize; j++) {
                dW2[i][j] += a1[i] * dz2[j];
            }
        }
        dz2.forEach((v, i) => db2[i] += v);

        const da1 = W2.map(row => row.reduce((sum, w, j) => sum + w * dz2[j], 0));
        const dz1 = da1.map((v, i) => z1[i] > 0 ? v : 0);

        for (let i = 0; i < vocabSize; i++) {
            for (let j = 0; j < hiddenSize; j++) {
                dW1[i][j] += x[i] * dz1[j];
            }
        }
        dz1.forEach((v, i) => db1[i] += v);
    });

    // Average and update
    const m = phrases.length;
    for (let i = 0; i < vocabSize; i++) {
        for (let j = 0; j < hiddenSize; j++) {
            W1[i][j] -= lr * dW1[i][j] / m;
        }
    }
    b1 = b1.map((v, i) => v - lr * db1[i] / m);

    for (let i = 0; i < hiddenSize; i++) {
        for (let j = 0; j < outputSize; j++) {
            W2[i][j] -= lr * dW2[i][j] / m;
        }
    }
    b2 = b2.map((v, i) => v - lr * db2[i] / m);

    greetingEpoch++;
    const loss = totalLoss / m;
    const accuracy = (correct / m * 100).toFixed(1);

    document.getElementById('greetingEpochVal').textContent = greetingEpoch;
    document.getElementById('greetingLossVal').textContent = loss.toFixed(4);
    document.getElementById('greetingAccVal').textContent = accuracy + '%';

    if (greetingEpoch >= 500 || (loss < 0.01 && accuracy === '100.0')) {
        stopGreetingTraining();
    }
}

function startGreetingTraining() {
    if (isGreetingTraining) return;
    isGreetingTraining = true;
    greetingInterval = setInterval(trainGreetingStep, 20);
}

function stopGreetingTraining() {
    isGreetingTraining = false;
    if (greetingInterval) clearInterval(greetingInterval);
}

function resetGreetingNetwork() {
    stopGreetingTraining();
    initGreetingNetwork();
    document.getElementById('greetingEpochVal').textContent = '0';
    document.getElementById('greetingLossVal').textContent = '0.00';
    document.getElementById('greetingAccVal').textContent = '0%';
    document.getElementById('greetingResult').style.display = 'none';
}

function testGreeting() {
    const input = document.getElementById('greetingInput').value.trim();
    if (!input) return;

    const result = classifyGreeting(input);
    document.getElementById('greetingResult').style.display = 'block';
    document.getElementById('greetingCategory').textContent =
        `Category: ${result.category.toUpperCase()}`;
    document.getElementById('greetingConfidence').textContent =
        `Confidence: ${result.confidence}%`;
}

// ============================================================
// Chat Demo
// ============================================================
function sendChat() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    const chatLog = document.getElementById('chatLog');

    // User message
    chatLog.innerHTML += `
        <div style="margin-bottom: 10px;">
            <span style="color: #555;">You:</span>
            <span style="color: #fff; margin-left: 10px;">${message}</span>
        </div>
    `;

    // AI response
    const response = getAIResponse(message);
    chatLog.innerHTML += `
        <div style="margin-bottom: 10px;">
            <span style="color: #666;">AI:</span>
            <span style="color: #888; margin-left: 10px;">${response}</span>
        </div>
    `;

    chatLog.scrollTop = chatLog.scrollHeight;
    input.value = '';
}

addTrainingLog('System initialized. Click "Start Training" to begin.');

