# Gradient Descent

## What is Gradient Descent?

Gradient Descent is the optimization algorithm that allows AI models to learn. It finds the best parameters by iteratively moving in the direction that reduces the error.

## The Intuition

Imagine you're blindfolded on a hilly terrain and need to find the lowest valley:

1. Feel the slope under your feet (calculate gradient)
2. Take a step downhill (update parameters)
3. Repeat until you reach the bottom (minimum loss)

```
        Loss
          │
          │    ●  ← Start here (random parameters)
          │     \
          │      \
          │       ●  ← Step 1
          │        \
          │         ●  ← Step 2
          │          \
          │           ●  ← Minimum (optimal parameters)
          └────────────────── Parameter Value
```

## The Mathematics

### Gradient

The **gradient** is the derivative of the loss function with respect to each parameter. It tells us:
- **Direction**: Which way is uphill
- **Magnitude**: How steep the slope is

### Update Rule

```
new_parameter = old_parameter - learning_rate × gradient
```

We subtract because we want to go **opposite** to the gradient (downhill).

## Learning Rate

The **learning rate** (α) controls step size:

```
Too Small (α = 0.0001):        Too Large (α = 1.0):         Just Right (α = 0.01):
        │                              │                            │
        ● → ● → ● → ● → ●             ●                            ●
        (very slow)                   / \                            \
                                     ●   ●                            ●
                                    /     \                            \
                                   ●       ●                            ●
                                  (overshoots)                         (converges)
```

## Types of Gradient Descent

### 1. Batch Gradient Descent
Uses entire dataset for each update.
```python
for epoch in range(epochs):
    gradient = compute_gradient(all_data)
    parameters -= learning_rate * gradient
```
- Stable but slow
- Good for small datasets

### 2. Stochastic Gradient Descent (SGD)
Uses one example at a time.
```python
for epoch in range(epochs):
    for example in dataset:
        gradient = compute_gradient(example)
        parameters -= learning_rate * gradient
```
- Fast but noisy
- Can escape local minima

### 3. Mini-batch Gradient Descent
Uses small batches (most common).
```python
for epoch in range(epochs):
    for batch in create_batches(dataset, batch_size=32):
        gradient = compute_gradient(batch)
        parameters -= learning_rate * gradient
```
- Best of both worlds
- Used in practice

## Derivation for Linear Regression

For model `y = wx + b` with Mean Squared Error loss:

```
Loss = (1/n) × Σ(prediction - actual)²
     = (1/n) × Σ(wx + b - y)²
```

**Gradient with respect to weight (w):**
```
∂Loss/∂w = (2/n) × Σ(wx + b - y) × x
```

**Gradient with respect to bias (b):**
```
∂Loss/∂b = (2/n) × Σ(wx + b - y)
```

## Code Implementation

```python
import numpy as np

def gradient_descent(X, Y, learning_rate=0.01, epochs=1000):
    weight = 0.0
    bias = 0.0
    n = len(X)
    
    for epoch in range(epochs):
        # Predictions
        predictions = weight * X + bias
        
        # Errors
        errors = predictions - Y
        
        # Gradients
        weight_gradient = (2/n) * np.sum(errors * X)
        bias_gradient = (2/n) * np.sum(errors)
        
        # Update parameters
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        
        # Track progress
        loss = np.mean(errors ** 2)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f}, w={weight:.3f}, b={bias:.3f}")
    
    return weight, bias

# Train
X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 5, 7, 9, 11])
w, b = gradient_descent(X, Y)
print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
```

## Advanced Optimizers

Modern deep learning uses improved versions:

| Optimizer | Key Feature |
|-----------|-------------|
| **Momentum** | Adds velocity to updates |
| **RMSprop** | Adapts learning rate per parameter |
| **Adam** | Combines Momentum + RMSprop |

### Adam (Most Popular)
```
m = β1 × m + (1-β1) × gradient        # Momentum
v = β2 × v + (1-β2) × gradient²       # Velocity
parameter -= lr × m / (√v + ε)        # Update
```

## Common Problems

### 1. Local Minima
Getting stuck in a suboptimal valley.
**Solution**: Momentum, random restarts

### 2. Saddle Points
Flat regions where gradient is zero.
**Solution**: Adam optimizer

### 3. Vanishing Gradients
Gradients become too small.
**Solution**: Better activation functions (ReLU)

## Key Takeaways

1. Gradient descent iteratively minimizes loss
2. Learning rate is crucial - not too big, not too small
3. Mini-batch is the standard approach
4. Modern optimizers (Adam) handle most issues automatically

## Next Steps

Continue to [Neural Networks](./04-neural-networks.md) to see gradient descent in action.

