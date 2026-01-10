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

# Training data: Expanded with more categories and phrases
greetings_data = {
    # Greetings
    "hello": "greeting",
    "hi": "greeting",
    "hey": "greeting",
    "good morning": "greeting",
    "good evening": "greeting",
    "good afternoon": "greeting",
    "greetings": "greeting",
    "howdy": "greeting",
    "hi there": "greeting",
    "hello there": "greeting",
    "hey there": "greeting",
    "welcome": "greeting",

    # Farewells
    "bye": "farewell",
    "goodbye": "farewell",
    "see you": "farewell",
    "later": "farewell",
    "see you later": "farewell",
    "take care": "farewell",
    "good night": "farewell",
    "farewell": "farewell",
    "bye bye": "farewell",
    "catch you later": "farewell",
    "until next time": "farewell",

    # Gratitude
    "thanks": "gratitude",
    "thank you": "gratitude",
    "appreciate": "gratitude",
    "thank you so much": "gratitude",
    "thanks a lot": "gratitude",
    "many thanks": "gratitude",
    "grateful": "gratitude",
    "much appreciated": "gratitude",
    "thanks for help": "gratitude",

    # Questions
    "how are you": "question",
    "whats up": "question",
    "how do you do": "question",
    "how is it going": "question",
    "what is new": "question",
    "how have you been": "question",
    "are you okay": "question",
    "you good": "question",

    # Apologies
    "sorry": "apology",
    "i am sorry": "apology",
    "my apologies": "apology",
    "excuse me": "apology",
    "pardon me": "apology",
    "forgive me": "apology",
    "my bad": "apology",
    "apologize": "apology",

    # Requests
    "please help": "request",
    "can you help": "request",
    "i need help": "request",
    "help me": "request",
    "could you": "request",
    "would you": "request",
    "can you": "request",
    "please": "request",

    # Affirmations
    "yes": "affirmation",
    "yeah": "affirmation",
    "yep": "affirmation",
    "sure": "affirmation",
    "okay": "affirmation",
    "alright": "affirmation",
    "absolutely": "affirmation",
    "definitely": "affirmation",
    "of course": "affirmation",

    # Negations
    "no": "negation",
    "nope": "negation",
    "not really": "negation",
    "never": "negation",
    "no way": "negation",
    "i dont think so": "negation",
    "negative": "negation",
}

categories = ["greeting", "farewell", "gratitude", "question", "apology", "request", "affirmation", "negation"]
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

# Deep Neural Network Parameters (3 hidden layers!)
np.random.seed(42)
hidden1_size = 32   # First hidden layer
hidden2_size = 16   # Second hidden layer
hidden3_size = 8    # Third hidden layer
output_size = len(categories)

# Initialize weights for deep network
W1 = np.random.randn(vocab_size, hidden1_size) * 0.3
b1 = np.zeros((1, hidden1_size))
W2 = np.random.randn(hidden1_size, hidden2_size) * 0.3
b2 = np.zeros((1, hidden2_size))
W3 = np.random.randn(hidden2_size, hidden3_size) * 0.3
b3 = np.zeros((1, hidden3_size))
W4 = np.random.randn(hidden3_size, output_size) * 0.3
b4 = np.zeros((1, output_size))

# Calculate total parameters
params_w1 = vocab_size * hidden1_size
params_w2 = hidden1_size * hidden2_size
params_w3 = hidden2_size * hidden3_size
params_w4 = hidden3_size * output_size
total_params = params_w1 + hidden1_size + params_w2 + hidden2_size + params_w3 + hidden3_size + params_w4 + output_size

print(f"  Total parameters: {total_params}")
print(f"    - W1: {vocab_size}x{hidden1_size} = {params_w1}")
print(f"    - b1: {hidden1_size}")
print(f"    - W2: {hidden1_size}x{hidden2_size} = {params_w2}")
print(f"    - b2: {hidden2_size}")
print(f"    - W3: {hidden2_size}x{hidden3_size} = {params_w3}")
print(f"    - b3: {hidden3_size}")
print(f"    - W4: {hidden3_size}x{output_size} = {params_w4}")
print(f"    - b4: {output_size}")

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training - More epochs for deeper network
learning_rate = 0.15
num_epochs = 5000
print(f"\n  Training deep neural network for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Forward pass through all 4 layers
    z1 = np.dot(X_train, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)
    z4 = np.dot(a3, W4) + b4
    a4 = softmax(z4)

    # Calculate loss (cross-entropy)
    loss = -np.mean(np.sum(Y_onehot * np.log(a4 + 1e-8), axis=1))

    # Backward pass through all layers
    m = X_train.shape[0]

    # Output layer gradients
    dz4 = a4 - Y_onehot
    dW4 = np.dot(a3.T, dz4) / m
    db4 = np.sum(dz4, axis=0, keepdims=True) / m

    # Hidden layer 3 gradients
    da3 = np.dot(dz4, W4.T)
    dz3 = da3 * relu_derivative(z3)
    dW3 = np.dot(a2.T, dz3) / m
    db3 = np.sum(dz3, axis=0, keepdims=True) / m

    # Hidden layer 2 gradients
    da2 = np.dot(dz3, W3.T)
    dz2 = da2 * relu_derivative(z2)
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    # Hidden layer 1 gradients
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X_train.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # Update all weights
    W4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        predictions = np.argmax(a4, axis=1)
        accuracy = np.mean(predictions == Y_train) * 100
        print(f"  Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.1f}%")

# Final accuracy
z1 = np.dot(X_train, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)
z3 = np.dot(a2, W3) + b3
a3 = relu(z3)
z4 = np.dot(a3, W4) + b4
a4 = softmax(z4)
predictions = np.argmax(a4, axis=1)
accuracy = np.mean(predictions == Y_train) * 100
print(f"\n  Final Accuracy: {accuracy:.1f}%")

# Test the greeting classifier (using deep network)
def classify_greeting(text):
    vec = text_to_vector(text.lower())
    # Forward pass through deep network
    z1 = np.dot(vec, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)
    z4 = np.dot(a3, W4) + b4
    a4 = softmax(z4)
    pred_idx = np.argmax(a4)
    confidence = a4[0, pred_idx] * 100
    return idx_to_cat[pred_idx], confidence

print("\n  Testing on various inputs:")
test_phrases = [
    "hello", "hi there", "good morning",      # greetings
    "bye", "see you later", "take care",      # farewells
    "thanks", "thank you so much",            # gratitude
    "how are you", "whats up",                # questions
    "sorry", "my apologies",                  # apologies
    "please help", "can you help",            # requests
    "yes", "absolutely",                      # affirmations
    "no", "never"                             # negations
]
for phrase in test_phrases:
    category, conf = classify_greeting(phrase)
    print(f"    '{phrase}' -> {category} ({conf:.1f}%)")


# ============================================================
# EXAMPLE 3: GREETING RESPONSE GENERATOR
# ============================================================
print("\n" + "=" * 60)
print("[EXAMPLE 3] Greeting Response Generator")
print("-" * 60)

response_data = {
    # Greetings
    "hello": "Hi there! How can I help you?",
    "hi": "Hello! Nice to meet you.",
    "hey": "Hey! What's going on?",
    "good morning": "Good morning! Hope you have a great day.",
    "good evening": "Good evening! How was your day?",
    "good afternoon": "Good afternoon! How can I assist you?",
    "howdy": "Howdy partner! What brings you here?",

    # Farewells
    "bye": "Goodbye! Take care.",
    "goodbye": "See you later! Have a nice day.",
    "see you": "See you soon! Take care.",
    "take care": "You too! Stay safe.",
    "good night": "Good night! Sweet dreams.",

    # Gratitude
    "thanks": "You're welcome!",
    "thank you": "My pleasure! Happy to help.",
    "thank you so much": "Anytime! I'm glad I could help.",

    # Questions
    "how are you": "I'm doing great, thanks for asking!",
    "whats up": "Not much, just here to help! What about you?",
    "how is it going": "It's going well! How can I assist?",

    # Apologies
    "sorry": "No worries at all!",
    "my apologies": "That's perfectly fine, no need to apologize.",

    # Requests
    "please help": "Of course! What do you need help with?",
    "help me": "I'm here to help! What's the problem?",

    # Affirmations
    "yes": "Great! Let's proceed then.",
    "okay": "Alright! What's next?",

    # Negations
    "no": "No problem! Let me know if you change your mind.",
    "never": "That's okay! Is there something else I can help with?",
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
        "apology": "No worries at all! It's perfectly fine.",
        "request": "Of course! I'm here to help. What do you need?",
        "affirmation": "Perfect! Let's move forward.",
        "negation": "That's okay! Let me know if you need anything else.",
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
        print("\n\n Server stopped. Goodbye!")
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
        print("\n Goodbye! Run 'python3 Ai-learining.py' again to launch the web visualizer.")

