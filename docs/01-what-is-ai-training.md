# What is AI Training?

## Introduction

AI training is the process of teaching a machine learning model to make accurate predictions by showing it examples. Think of it like teaching a child to recognize objects - you show them many examples until they understand the pattern.

## The Core Concept

At its heart, AI training is about finding the right **parameters** (numbers) that allow a mathematical function to map inputs to outputs correctly.

```
Input → [Model with Parameters] → Output
```

## Key Components

### 1. Training Data

Training data consists of input-output pairs that the model learns from.

```
Example: Teaching AI to predict house prices

Input Features:          Output:
- Size: 1500 sq ft      → Price: $300,000
- Size: 2000 sq ft      → Price: $400,000
- Size: 2500 sq ft      → Price: $500,000
```

### 2. Model

The model is a mathematical function with adjustable parameters. In our simple example:

```
y = weight × x + bias
```

- **weight** and **bias** are parameters we need to learn
- **x** is the input
- **y** is the predicted output

### 3. Loss Function

The loss function measures how wrong the model's predictions are.

```
prediction = model(input)
error = prediction - actual_output
loss = error²
```

A lower loss means better predictions.

### 4. Optimization

The optimization algorithm adjusts parameters to minimize the loss. The most common is **Gradient Descent**.

## The Training Loop

```
1. FORWARD PASS
   - Feed input through the model
   - Get a prediction

2. CALCULATE LOSS
   - Compare prediction to actual output
   - Compute the error

3. BACKWARD PASS
   - Calculate gradients (direction to adjust parameters)

4. UPDATE PARAMETERS
   - Nudge parameters in the direction that reduces error

5. REPEAT
   - Go back to step 1 with the next example
```

## Visual Representation

```
    Training Data
         │
         ▼
    ┌─────────┐
    │  Model  │◄──── Parameters (weights, biases)
    └────┬────┘
         │
         ▼
    Predictions
         │
         ▼
    ┌─────────┐
    │  Loss   │◄──── Compare with actual outputs
    └────┬────┘
         │
         ▼
    Gradients
         │
         ▼
    Update Parameters
         │
         └──────► Repeat until loss is minimized
```

## Simple Python Example

```python
# Training data: y = 2x + 1
X = [1, 2, 3, 4, 5]
Y = [3, 5, 7, 9, 11]

# Initialize parameters
weight = 0.0
bias = 0.0
learning_rate = 0.01

# Training loop
for epoch in range(1000):
    for x, y_actual in zip(X, Y):
        # Forward pass
        prediction = weight * x + bias
        
        # Calculate loss
        error = prediction - y_actual
        
        # Calculate gradients
        weight_grad = 2 * error * x
        bias_grad = 2 * error
        
        # Update parameters
        weight -= learning_rate * weight_grad
        bias -= learning_rate * bias_grad

print(f"Learned: y = {weight:.2f}x + {bias:.2f}")
# Output: Learned: y = 2.00x + 1.00
```

## Why Does This Work?

The model starts with random guesses for weight and bias. Each training step:

1. The model makes a prediction (probably wrong at first)
2. We measure how wrong it was (loss)
3. We calculate which direction to adjust parameters (gradients)
4. We make small adjustments to reduce the error
5. Over many iterations, the parameters converge to optimal values

## Key Takeaways

- Training is an iterative process of prediction and correction
- The model learns by adjusting parameters to minimize loss
- More data generally leads to better learning
- The learning rate controls how fast parameters change

## Next Steps

Continue to [Epochs and Iterations](./02-epochs-and-iterations.md) to learn about training cycles.

