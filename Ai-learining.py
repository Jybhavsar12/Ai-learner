
import numpy as np

# ========== 1. CREATE TRAINING DATA ==========
# We want the AI to learn: y = 2x + 1
X = np.array([1, 2, 3, 4, 5])           # inputs
Y = np.array([3, 5, 7, 9, 11])          # outputs (2*x + 1)

# ========== 2. INITIALIZE MODEL ==========
# Our model: y = weight * x + bias
# Start with random guesses
weight = 0.0
bias = 0.0
learning_rate = 0.01

# ========== 3. TRAINING LOOP ==========
print("Training started...\n")

for epoch in range(1000):
    # Forward pass: make predictions
    predictions = weight * X + bias
    
    # Calculate error (Mean Squared Error)
    error = predictions - Y
    loss = np.mean(error ** 2)
    
    # Backward pass: calculate gradients
    weight_gradient = np.mean(2 * error * X)
    bias_gradient = np.mean(2 * error)
    
    # Update weights (gradient descent)
    weight = weight - learning_rate * weight_gradient
    bias = bias - learning_rate * bias_gradient
    
    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Weight = {weight:.4f}, Bias = {bias:.4f}")

# ========== 4. RESULTS ==========
print(f"\n✅ Training complete!")
print(f"Learned: y = {weight:.2f}x + {bias:.2f}")
print(f"Actual:  y = 2.00x + 1.00")

# ========== 5. TEST THE MODEL ==========
print("\n🧪 Testing on new data:")
test_x = 10
prediction = weight * test_x + bias
actual = 2 * test_x + 1
print(f"Input: {test_x}")
print(f"Predicted: {prediction:.2f}")
print(f"Actual: {actual}")

