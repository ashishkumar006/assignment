import torch
from model import CompactMNIST  # Use the actual model class name
from train import evaluate_model  # Ensure this function is defined in train.py

def check_model():
    model = CompactMNIST()  # Initialize your model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check number of parameters
    if num_params >= 25000:
        raise ValueError(f"Number of parameters is {num_params}, which exceeds 25,000.")

    # Evaluate the model to get accuracy
    accuracy = evaluate_model(model)  # Implement this function to return accuracy
    
    # Check accuracy
    if accuracy <= 95:  # Ensure accuracy is greater than 95%
        raise ValueError(f"Accuracy is {accuracy:.2f}%, which is less than or equal to 95%.")

    print("Model checks passed: Parameters < 25,000 and Accuracy > 95%.")

if __name__ == "__main__":
    check_model()
