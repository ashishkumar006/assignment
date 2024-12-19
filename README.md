# Compact MNIST Classifier

[![Model Tests](https://github.com/{username}/{repository}/actions/workflows/model_test.yml/badge.svg)](https://github.com/{username}/{repository}/actions/workflows/model_test.yml)

This repository contains a PyTorch implementation of a compact MNIST classifier that achieves >95% accuracy in a single epoch while using less than 25,000 parameters.

## Testing Strategy
- Automated testing on GitHub Actions
- Multiple training attempts to ensure reliability
- Deterministic training for reproducibility
- Parameter count verification
- Accuracy threshold verification (>95%)

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- tqdm

## Usage
```bash
pip install -r requirements.txt
python train.py
```

## Model Details
- Parameters: 19,570
- Target Accuracy: >95%
- Training Time: Single epoch
- Architecture: CNN with residual connections

## License
MIT