"""
Utility functions
"""
import torch
import numpy as np

def accuracy(output, labels, mask):
    """Calculate accuracy by label and model's output"""
    
    output = torch.argmax(output, dim=1).numpy()
    labels = torch.argmax(labels, dim=1).numpy()
    correct = output == labels

    print(correct)
    
    # masking accuracy
    mask = mask.float().numpy()
    TP = np.sum(correct * mask)

    acc = TP / np.sum(mask)

    return acc
