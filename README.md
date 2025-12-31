# AI Training Visualizer

A hands-on educational project that demonstrates how artificial intelligence learns through interactive visualizations and comprehensive documentation.

## Overview

This project provides:
- A **Python implementation** of a simple AI training algorithm
- An **interactive web visualizer** to watch AI learn in real-time
- **Comprehensive documentation** explaining AI concepts from basics to advanced

## Quick Start

### Python Version
```bash
cd Ai-learner
pip install numpy
python Ai-learining.py
```

### Web Visualizer
Open `index.html` in your browser and click "Start Training" to watch the AI learn.

## What You'll Learn

| Concept | Description |
|---------|-------------|
| Training Data | How AI learns from examples |
| Model Parameters | Weights and biases that get adjusted |
| Loss Function | How we measure prediction errors |
| Gradient Descent | The optimization algorithm |
| Epochs | Training iterations through data |
| Neural Networks | Building blocks of deep learning |
| ChatGPT | How large language models are trained |

## Project Structure

```
Ai-learner/
├── README.md              # This file
├── Ai-learining.py        # Python training implementation
├── index.html             # Web visualizer interface
├── training.js            # JavaScript training logic
└── docs/
    ├── 01-what-is-ai-training.md
    ├── 02-epochs-and-iterations.md
    ├── 03-gradient-descent.md
    ├── 04-neural-networks.md
    └── 05-how-chatgpt-works.md
```

## The Training Task

The AI learns to predict the function: **y = 2x + 1**

Given training data:
| Input (x) | Output (y) |
|-----------|------------|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |
| 5 | 11 |

The model discovers that `weight = 2` and `bias = 1` through gradient descent.

## Features

### Web Visualizer
- Real-time loss curve visualization
- Model prediction line overlay
- Live statistics dashboard
- Step-through training mode
- Question logging system

### Documentation
- Beginner-friendly explanations
- Mathematical foundations
- Code examples
- Diagrams and visual aids

## Technologies Used

- Python 3 with NumPy
- Vanilla JavaScript
- Chart.js for visualizations
- HTML5 / CSS3

## Documentation

See the [docs](./docs/) folder for detailed explanations:

1. [What is AI Training?](./docs/01-what-is-ai-training.md)
2. [Epochs and Iterations](./docs/02-epochs-and-iterations.md)
3. [Gradient Descent](./docs/03-gradient-descent.md)
4. [Neural Networks](./docs/04-neural-networks.md)
5. [How ChatGPT Works](./docs/05-how-chatgpt-works.md)

## License

MIT License - Feel free to use this for learning and teaching.

## Author

Created as an educational resource for understanding AI fundamentals.

