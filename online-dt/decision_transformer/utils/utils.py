import numpy as np
import torch

def truncate_insulin(action, threshold=0.1):
    """
    Truncates insulin amounts close to zero to exactly zero.
    
    Args:
        action (np.ndarray or torch.Tensor): Predicted insulin action (scalar or array).
        threshold (float): Threshold below which insulin is set to zero (default: 0.1).
    
    Returns:
        Truncated action (same type as input).
    """
    if isinstance(action, torch.Tensor):
        return torch.where(action.abs() < threshold, torch.zeros_like(action), action)
    else:  # Assume numpy array
        return np.where(np.abs(action) < threshold, 0.0, action)
