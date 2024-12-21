import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import CompactMNIST
from train import train_one_epoch, count_parameters
import pytest

def test_parameter_count():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25,000"

def test_accuracy():
    try:
        model, accuracy = train_one_epoch()
        assert accuracy > 94.5, f"Accuracy {accuracy:.2f}% is too low"
    except Exception as e:
        pytest.fail(f"Training failed with error: {str(e)}")