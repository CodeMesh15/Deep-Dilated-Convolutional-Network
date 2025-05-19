from utils import get_data_loaders, accuracy
import torch

def evaluate(model):
    _, test_loader = get_data_loaders()
    model.eval()
    acc = accuracy(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")
