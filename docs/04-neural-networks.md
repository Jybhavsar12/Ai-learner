# Neural Networks

## What is a Neural Network?

A neural network is a series of connected layers that transform input data into output predictions. Each layer consists of **neurons** that apply weights, biases, and activation functions.

## From Linear to Neural

Our simple model: `y = wx + b` (1 neuron)

Neural network: Many neurons connected in layers

```
Simple Model:              Neural Network:
                           
Input → [w,b] → Output     Input → [Layer1] → [Layer2] → [Layer3] → Output
                                     ↓           ↓           ↓
                                  neurons     neurons     neurons
```

## Architecture of a Neuron

Each neuron performs:

```
output = activation(Σ(weight × input) + bias)
```

```
    Input 1 ──(w1)──┐
                    │
    Input 2 ──(w2)──┼──► [Σ + bias] ──► [Activation] ──► Output
                    │
    Input 3 ──(w3)──┘
```

## Activation Functions

Activation functions introduce non-linearity, allowing networks to learn complex patterns.

### Common Activation Functions

```
ReLU:               Sigmoid:            Tanh:
   │    /              │   ___             │    ___
   │   /               │  /               ─┼───/
───┼──/            ───┼─/              ___/│
   │                   │                    │
   
f(x) = max(0,x)    f(x) = 1/(1+e^-x)   f(x) = tanh(x)
```

| Function | Range | Use Case |
|----------|-------|----------|
| ReLU | [0, ∞) | Hidden layers |
| Sigmoid | (0, 1) | Binary classification |
| Softmax | (0, 1) | Multi-class classification |
| Tanh | (-1, 1) | Hidden layers |

## Network Architecture

```
INPUT LAYER          HIDDEN LAYERS           OUTPUT LAYER
(features)           (learn patterns)        (predictions)
    
    ○                    ○    ○                   ○
    ○ ──────────────►    ○    ○  ──────────────►  ○
    ○                    ○    ○                   ○
    ○                    ○    ○
    
  4 inputs            2 hidden layers          2 outputs
                      (4 neurons each)
```

## Forward Propagation

Data flows forward through the network:

```python
def forward(x, weights, biases):
    # Layer 1
    z1 = np.dot(x, weights[0]) + biases[0]
    a1 = relu(z1)
    
    # Layer 2
    z2 = np.dot(a1, weights[1]) + biases[1]
    a2 = relu(z2)
    
    # Output layer
    z3 = np.dot(a2, weights[2]) + biases[2]
    output = softmax(z3)
    
    return output
```

## Backpropagation

The algorithm for computing gradients through all layers.

### Chain Rule
To find how a weight affects the final loss, we chain partial derivatives:

```
∂Loss/∂w1 = ∂Loss/∂output × ∂output/∂a2 × ∂a2/∂z2 × ∂z2/∂a1 × ∂a1/∂z1 × ∂z1/∂w1
```

### Backprop Flow

```
Forward:  Input → Layer1 → Layer2 → Layer3 → Output → Loss
                                                        │
Backward: Update ← Grad1 ← Grad2 ← Grad3 ← ───────────┘
```

## Simple Neural Network Code

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            self.activations.append(relu(z))
        return self.activations[-1]
    
    def train(self, X, y, learning_rate=0.01):
        # Forward
        output = self.forward(X)
        
        # Backward
        error = output - y
        for i in range(len(self.weights) - 1, -1, -1):
            gradient = error * relu_derivative(self.activations[i+1])
            self.weights[i] -= learning_rate * np.dot(
                self.activations[i].T, gradient
            )
            self.biases[i] -= learning_rate * np.sum(gradient, axis=0)
            error = np.dot(gradient, self.weights[i].T)
```

## Deep Learning

"Deep" means many layers. More layers can learn more complex patterns:

| Depth | Learns |
|-------|--------|
| 1 layer | Linear patterns |
| 2-3 layers | Simple non-linear patterns |
| 10+ layers | Complex hierarchical features |
| 100+ layers | State-of-the-art (ResNet, GPT) |

## Key Concepts

### Overfitting Prevention
- **Dropout**: Randomly disable neurons during training
- **Regularization**: Penalize large weights
- **Early stopping**: Stop when validation loss increases

### Batch Normalization
Normalize activations between layers for faster, more stable training.

## Key Takeaways

1. Neural networks stack layers of neurons
2. Activation functions enable non-linear learning
3. Backpropagation computes gradients efficiently
4. Deeper networks learn more complex patterns
5. Regularization prevents overfitting

## Next Steps

Continue to [How ChatGPT Works](./05-how-chatgpt-works.md) to see neural networks at scale.

