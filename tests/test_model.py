import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import CompactMNIST
from train import train_one_epoch, count_parameters

def test_parameter_count():
    model = CompactMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, which exceeds the limit of 25,000"

def test_accuracy():
    try:
        accuracy = train_one_epoch()
        assert accuracy > 95, f"Model accuracy {accuracy}% is below the required 95%"
    except RuntimeError as e:
        # Handle case where CUDA is not available
        if "CUDA" in str(e):
            print("Warning: CUDA not available, skipping accuracy test")
            return
        raise e