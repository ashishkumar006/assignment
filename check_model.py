import torch
import logging
from model import CompactMNIST
from train import train_one_epoch, evaluate_model

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_model(param_threshold=25000, accuracy_threshold=95):
    """
    Check the model's parameters and accuracy.

    Args:
        param_threshold (int): Maximum allowed number of parameters.
        accuracy_threshold (float): Minimum required accuracy.

    Raises:
        ValueError: If the model exceeds the parameter threshold or does not meet the accuracy threshold.
    """
    # Initialize and train model
    model = CompactMNIST()
    logging.info("Model initialized. Training model...")
    
    # Train the model
    train_accuracy = train_one_epoch()
    logging.info(f"Training completed. Training accuracy: {train_accuracy:.2f}%")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {num_params}")

    # Check number of parameters
    if num_params >= param_threshold:
        raise ValueError(f"Number of parameters is {num_params}, which exceeds {param_threshold}.")

    # Evaluate the model
    logging.info("Evaluating model...")
    accuracy = evaluate_model(model)
    logging.info(f"Evaluation accuracy: {accuracy:.2f}%")
    
    # Check accuracy
    if accuracy <= accuracy_threshold:
        raise ValueError(f"Accuracy is {accuracy:.2f}%, which is less than or equal to {accuracy_threshold}%.")

    logging.info(f"All checks passed! Parameters: {num_params} < {param_threshold}, Accuracy: {accuracy:.2f}% > {accuracy_threshold}%")

if __name__ == "__main__":
    try:
        check_model()
    except Exception as e:
        logging.error(f"Check failed: {str(e)}")
        raise
