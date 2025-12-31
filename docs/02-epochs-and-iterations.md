# Epochs and Iterations

## What is an Epoch?

An **epoch** is one complete pass through the entire training dataset. If you have 1000 training examples and you show all 1000 to the model once, that's one epoch.

```
Dataset: [example1, example2, example3, example4, example5]

Epoch 1: Model sees all 5 examples → Updates parameters
Epoch 2: Model sees all 5 examples again → Updates parameters
Epoch 3: Model sees all 5 examples again → Updates parameters
...
```

## What is an Iteration (or Step)?

An **iteration** (also called a step or batch) is one update of the model's parameters. Depending on the training setup:

- **Stochastic Gradient Descent (SGD)**: 1 iteration = 1 example
- **Mini-batch**: 1 iteration = batch of examples (e.g., 32)
- **Batch**: 1 iteration = entire dataset

## Relationship Between Epochs and Iterations

```
Total Iterations = (Number of Examples / Batch Size) × Number of Epochs
```

**Example:**
- Dataset: 1000 examples
- Batch size: 100
- Epochs: 10

```
Iterations per epoch = 1000 / 100 = 10
Total iterations = 10 × 10 = 100 parameter updates
```

## Why Multiple Epochs?

One pass through the data is rarely enough. Multiple epochs allow the model to:

1. **Refine parameters** - Each pass makes small improvements
2. **Learn complex patterns** - Some patterns require multiple exposures
3. **Reduce noise** - Averaging over many passes reduces random fluctuations

### Analogy: Studying for an Exam

```
Epoch 1: Read the textbook once
         → You remember some things, forget others

Epoch 2: Read the textbook again
         → Concepts start connecting

Epoch 3: Read the textbook again
         → Deeper understanding forms

...

Epoch N: Material is fully learned
```

## Visualizing Training Over Epochs

```
Loss
  │
  │█
  │██
  │███
  │████
  │█████
  │██████
  │███████
  │████████████████████████████ (plateaus)
  └─────────────────────────────────── Epochs
   0    100   200   300   400   500
```

- **Early epochs**: Rapid loss decrease
- **Middle epochs**: Gradual improvement
- **Later epochs**: Diminishing returns (plateau)

## Batch Size Trade-offs

| Batch Size | Pros | Cons |
|------------|------|------|
| Small (1-32) | More updates, can escape local minima | Noisy gradients, slower |
| Medium (32-256) | Good balance | - |
| Large (256+) | Stable gradients, faster per epoch | May converge to sharp minima |

## Code Example: Tracking Epochs

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 5, 7, 9, 11])

weight, bias = 0.0, 0.0
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Forward pass (all data at once)
    predictions = weight * X + bias
    
    # Calculate loss
    loss = np.mean((predictions - Y) ** 2)
    
    # Backward pass
    weight_grad = np.mean(2 * (predictions - Y) * X)
    bias_grad = np.mean(2 * (predictions - Y))
    
    # Update
    weight -= learning_rate * weight_grad
    bias -= learning_rate * bias_grad
    
    # Log every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

Output:
```
Epoch 0: Loss = 57.0000
Epoch 200: Loss = 0.0081
Epoch 400: Loss = 0.0021
Epoch 600: Loss = 0.0005
Epoch 800: Loss = 0.0001
```

## How Many Epochs to Train?

There's no universal answer. Consider:

### Early Stopping
Stop when validation loss stops improving:
```
if validation_loss > previous_validation_loss:
    patience_counter += 1
    if patience_counter >= patience_limit:
        stop_training()
```

### Learning Curves
Plot training and validation loss:
- **Both decreasing**: Keep training
- **Training decreasing, validation increasing**: Overfitting - stop
- **Both plateaued**: Model has converged

## Overfitting vs Underfitting

```
         Loss
           │    Underfitting    │    Good Fit    │   Overfitting
           │                    │                │
Training   │    ████████████    │    ██████      │   ██
Validation │    ████████████    │    ██████      │   ██████████
           │                    │                │
           └────────────────────┴────────────────┴──────────────
                Too few epochs      Right amount    Too many epochs
```

## Key Takeaways

1. **Epoch** = One complete pass through all training data
2. **Iteration** = One parameter update
3. Multiple epochs refine the model gradually
4. Too few epochs → Underfitting
5. Too many epochs → Overfitting
6. Use validation data to determine when to stop

## Next Steps

Continue to [Gradient Descent](./03-gradient-descent.md) to understand the optimization algorithm.

