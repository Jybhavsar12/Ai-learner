import numpy as np
import webbrowser
import os
import http.server
import socketserver
import threading

print("=" * 60)
print("           AI TRAINING EXAMPLES")
print("=" * 60)

# ============================================================
# EXAMPLE 1: SIMPLE LINEAR REGRESSION (y = 2x + 1)
# ============================================================
print("\n[EXAMPLE 1] Linear Regression: Learning y = 2x + 1")
print("-" * 60)

X_linear = np.array([1, 2, 3, 4, 5])
Y_linear = np.array([3, 5, 7, 9, 11])

weight = 0.0
bias = 0.0
learning_rate = 0.01

for epoch in range(1000):
    predictions = weight * X_linear + bias
    error = predictions - Y_linear
    loss = np.mean(error ** 2)

    weight -= learning_rate * np.mean(2 * error * X_linear)
    bias -= learning_rate * np.mean(2 * error)

    if epoch % 200 == 0:
        print(f"  Epoch {epoch}: Loss = {loss:.4f}, w = {weight:.3f}, b = {bias:.3f}")

print(f"\n  Result: Learned y = {weight:.2f}x + {bias:.2f}")
print(f"  Target: y = 2.00x + 1.00")


# ============================================================
# EXAMPLE 2: GREETING CLASSIFIER (Neural Network)
# ============================================================
print("\n" + "=" * 60)
print("[EXAMPLE 2] Greeting Classifier: Neural Network")
print("-" * 60)

# Training data: Greetings and their categories
greetings_data = {
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
    "how do you do": "question",
}

categories = ["greeting", "farewell", "gratitude", "question"]
cat_to_idx = {cat: i for i, cat in enumerate(categories)}
idx_to_cat = {i: cat for i, cat in enumerate(categories)}

# Create vocabulary
all_words = set()
for phrase in greetings_data.keys():
    for word in phrase.split():
        all_words.add(word)
vocab = sorted(list(all_words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

print(f"  Vocabulary size: {vocab_size} words")
print(f"  Categories: {categories}")
print(f"  Training samples: {len(greetings_data)}")

# Convert text to vectors (bag of words)
def text_to_vector(text):
    vector = np.zeros(vocab_size)
    for word in text.split():
        if word in word_to_idx:
            vector[word_to_idx[word]] = 1
    return vector

# Create training data
X_train = np.array([text_to_vector(text) for text in greetings_data.keys()])
Y_train = np.array([cat_to_idx[cat] for cat in greetings_data.values()])

# One-hot encode labels
Y_onehot = np.zeros((len(Y_train), len(categories)))
for i, y in enumerate(Y_train):
    Y_onehot[i, y] = 1

# Neural Network Parameters
np.random.seed(42)
hidden_size = 8
output_size = len(categories)

# Initialize weights (MORE PARAMETERS!)
W1 = np.random.randn(vocab_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))

total_params = (vocab_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
print(f"  Total parameters: {total_params}")
print(f"    - W1: {vocab_size}x{hidden_size} = {vocab_size * hidden_size}")
print(f"    - b1: {hidden_size}")
print(f"    - W2: {hidden_size}x{output_size} = {hidden_size * output_size}")
print(f"    - b2: {output_size}")

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training
learning_rate = 0.1
print(f"\n  Training neural network...")

for epoch in range(2000):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # Calculate loss (cross-entropy)
    loss = -np.mean(np.sum(Y_onehot * np.log(a2 + 1e-8), axis=1))

    # Backward pass
    m = X_train.shape[0]
    dz2 = a2 - Y_onehot
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X_train.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 400 == 0:
        predictions = np.argmax(a2, axis=1)
        accuracy = np.mean(predictions == Y_train) * 100
        print(f"  Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.1f}%")

# Final accuracy
z1 = np.dot(X_train, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)
predictions = np.argmax(a2, axis=1)
accuracy = np.mean(predictions == Y_train) * 100
print(f"\n  Final Accuracy: {accuracy:.1f}%")

# Test the greeting classifier
def classify_greeting(text):
    vec = text_to_vector(text.lower())
    z1 = np.dot(vec, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    pred_idx = np.argmax(a2)
    confidence = a2[0, pred_idx] * 100
    return idx_to_cat[pred_idx], confidence

print("\n  Testing on new inputs:")
test_phrases = ["hello", "bye", "thanks", "how are you", "good morning", "see you"]
for phrase in test_phrases:
    category, conf = classify_greeting(phrase)
    print(f"    '{phrase}' -> {category} ({conf:.1f}% confidence)")


# ============================================================
# EXAMPLE 3: GREETING RESPONSE GENERATOR
# ============================================================
print("\n" + "=" * 60)
print("[EXAMPLE 3] Greeting Response Generator")
print("-" * 60)

response_data = {
    "hello": "Hi there! How can I help you?",
    "hi": "Hello! Nice to meet you.",
    "hey": "Hey! What's going on?",
    "good morning": "Good morning! Hope you have a great day.",
    "good evening": "Good evening! How was your day?",
    "bye": "Goodbye! Take care.",
    "goodbye": "See you later! Have a nice day.",
    "thanks": "You're welcome!",
    "thank you": "My pleasure! Happy to help.",
    "how are you": "I'm doing great, thanks for asking!",
}

print(f"  Learned {len(response_data)} response patterns")

def get_response(user_input):
    user_input = user_input.lower().strip()

    if user_input in response_data:
        return response_data[user_input]

    category, confidence = classify_greeting(user_input)

    default_responses = {
        "greeting": "Hello! How can I assist you today?",
        "farewell": "Goodbye! Have a wonderful day!",
        "gratitude": "You're welcome! Glad I could help.",
        "question": "I'm doing well! Thanks for asking.",
    }

    return default_responses.get(category, "I'm not sure how to respond to that.")

print("\n  Demo responses:")
print("  " + "-" * 40)

demo_inputs = ["hello", "how are you", "thanks", "bye"]
for inp in demo_inputs:
    response = get_response(inp)
    print(f"    You: {inp}")
    print(f"    AI:  {response}\n")


# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"""
Example 1 - Linear Regression:
  - Parameters: 2 (weight, bias)
  - Task: Learn y = 2x + 1
  - Method: Gradient Descent

Example 2 - Greeting Classifier:
  - Parameters: {total_params} (weights + biases)
  - Task: Classify text into categories
  - Method: Neural Network with Backpropagation

Example 3 - Response Generator:
  - Built on top of the classifier
  - Maps categories to appropriate responses
  - Demonstrates practical AI application

Key Concepts Demonstrated:
  - Forward propagation
  - Loss calculation
  - Backpropagation
  - Gradient descent optimization
  - Activation functions (ReLU, Softmax)
  - Multi-layer neural networks
""")

# ============================================================
# LAUNCH WEB VISUALIZER
# ============================================================
def launch_web_visualizer():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to script directory for serving files
    os.chdir(script_dir)

    PORT = 8080

    # Create a simple HTTP server
    handler = http.server.SimpleHTTPRequestHandler

    # Suppress server logs
    handler.log_message = lambda *args: None

    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            url = f"http://localhost:{PORT}/index.html"
            print("=" * 60)
            print("🌐 WEB VISUALIZER")
            print("=" * 60)
            print(f"\n  Server running at: http://localhost:{PORT}")
            print(f"  Opening browser...")
            print(f"\n  Press Ctrl+C to stop the server\n")
            print("=" * 60)

            # Open browser
            webbrowser.open(url)

            # Keep server running
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except OSError as e:
        if e.errno == 48:  # Port already in use
            print(f"\n⚠️  Port {PORT} is already in use.")
            print(f"  Opening browser anyway...")
            webbrowser.open(f"http://localhost:{PORT}/index.html")
        else:
            raise

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Would you like to launch the web visualizer? (y/n): ", end="")
    choice = input().strip().lower()

    if choice in ['y', 'yes', '']:
        launch_web_visualizer()
    else:
        print("\n👋 Goodbye! Run 'python3 Ai-learining.py' again to launch the web visualizer.")

