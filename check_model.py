import torch
import logging
from model import CompactMNIST
from train import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_model(param_threshold=25000, accuracy_threshold=95):
    """
    Check the model's parameters and accuracy.

    Args:
        param_threshold (int): Maximum allowed number of parameters.
        accuracy_threshold (float): Minimum required accuracy.

    Raises:
        ValueError: If the model exceeds the parameter threshold or does not meet the accuracy threshold.
    """
    model = CompactMNIST()  # Initialize your model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check number of parameters
    if num_params >= param_threshold:
        raise ValueError(f"Number of parameters is {num_params}, which exceeds {param_threshold}.")

    # Evaluate the model to get accuracy
    accuracy = evaluate_model(model)
    
    # Check accuracy
    if accuracy <= accuracy_threshold:
        raise ValueError(f"Accuracy is {accuracy:.2f}%, which is less than or equal to {accuracy_threshold}.")

    logging.info("Model checks passed: Parameters < %d and Accuracy > %.2f%%.", param_threshold, accuracy_threshold)

if __name__ == "__main__":
    check_model()
