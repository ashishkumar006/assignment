# Compact MNIST Classifier 🚀

[![Build Status](https://github.com/ashishkumar006/assignment/actions/workflows/model_test.yml/badge.svg)](https://github.com/ashishkumar006/assignment/actions/workflows/model_test.yml)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight PyTorch implementation of an MNIST classifier achieving **98.62%** accuracy in a single epoch with fewer than 25,000 parameters. Designed for efficiency and reproducibility.

## 🎯 Model Performance

| Metric | Value 
|--------|-------
| Parameters | 19,570 
| Training Accuracy | 95.80% 
| Test Accuracy | 98.62% 
| Training Time | 1 epoch 

## 📦 Requirements

```bash
torch==1.10.0+cpu
torchvision
numpy
tqdm
pytest
```

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/ashishkumar006/assignment.git
cd assignment

# Install dependencies
pip install -r requirements.txt

# Train and evaluate model
python train.py

# Run tests
pytest tests/
```

## 🔍 Testing Strategy

- ✅ Automated testing via GitHub Actions
- 📊 Parameter count verification (<25,000)
- 📈 Accuracy threshold verification (>95%)
- 🎲 Deterministic training with fixed seeds
- 💻 CPU-compatible implementation

## 🏗️ Model Architecture

- 🧠 Compact CNN design
- ⚡ Efficient parameter utilization
- 🔄 Residual connections
- 🎯 Optimized for single-epoch training

## ⚙️ Implementation Details

- 📈 AdamW optimizer
- 📉 Learning rate scheduling
- 🔗 Gradient clipping
- 🎲 Deterministic training
- 🖼️ No data augmentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Performance Visualization

```
Training Progress:
[====================] 100%
Final Accuracy: 98.62%
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📫 Contact

If you have any questions, feel free to open an issue or reach out directly.