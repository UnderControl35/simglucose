import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import pickle

def truncate_insulin(action, threshold=0.1):
    """
    Truncates insulin amounts close to zero to exactly zero and ensures non-negativity.
    
    Args:
        action (np.ndarray or torch.Tensor): Predicted insulin action (scalar or array).
        threshold (float): Threshold below which insulin is set to zero (default: 0.1).
    
    Returns:
        Truncated action (same type as input).
    """
    
    if isinstance(action, torch.Tensor):
        # Ensure non-negativity first
        action = torch.clamp(action, min=0.0)
        # Truncate near-zero values
        return torch.where(action.abs() < threshold, torch.zeros_like(action), action)
    else:  # Assume numpy array
        # Ensure non-negativity
        action = np.maximum(action, 0.0)
        # Truncate near-zero values
        return np.where(np.abs(action) < threshold, 0.0, action)
    

def analyze_glycemic_states(file_path, plot=False, hypo_threshold=70, hyper_threshold=180):
    """
    Analyze the fraction of time spent in different glycemic states from a list of glucose readings.
    
    Parameters:
    - file_path (str): Path to the .pkl file containing a list of glucose readings.
    - plot (bool): If True, display a bar plot of the results.
    - hypo_threshold (float): Threshold for hypoglycemia (default: 70 mg/dL).
    - hyper_threshold (float): Threshold for hyperglycemia (default: 180 mg/dL).
    
    Returns:
    - dict: Fractions of time spent in each glycemic state.
    """
    # Load the data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract glucose readings (flatten if nested)
    glucose_data = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (list, dict)):
                glucose_data.extend(item if isinstance(item, list) else item.get('glucose', []))
            else:
                glucose_data.append(item)
    else:
        raise ValueError("Expected a list, got type:", type(data))

    # Convert to Pandas Series
    glucose_series = pd.Series(glucose_data)

    # Categorize glycemic states
    def categorize_state(glucose):
        try:
            glucose = float(glucose)
            if glucose < hypo_threshold: return 'Hypoglycemia'
            elif glucose <= hyper_threshold: return 'Euglycemia'
            return 'Hyperglycemia'
        except (ValueError, TypeError):
            return 'Invalid'

    states = glucose_series.apply(categorize_state)
    states = states[states != 'Invalid']  # Filter invalid entries

    # Compute fractions
    fractions = (states.value_counts() / len(states)).to_dict()

    # Plot if requested
    if plot:
        pd.Series(fractions).plot(kind='bar', color=['red', 'green', 'orange'])
        plt.title('Fraction of Time in Glycemic States')
        plt.xlabel('Glycemic State')
        plt.ylabel('Fraction of Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return fractions

# Modified analyze_glycemic_states to work with a list directly
def analyze_glycemic_states_from_list(glucose_data, plot=False, hypo_threshold=70, hyper_threshold=180):
    """
    Analyze glycemic states from a list of glucose readings (no file loading).
    
    Parameters:
    - glucose_data (list): List of glucose readings.
    - plot (bool): If True, display a bar plot.
    - hypo_threshold (float): Hypoglycemia threshold (default: 70 mg/dL).
    - hyper_threshold (float): Hyperglycemia threshold (default: 180 mg/dL).
    
    Returns:
    - dict: Fractions of time in each glycemic state.
    """
    glucose_series = pd.Series(glucose_data)

    def categorize_state(glucose):
        try:
            glucose = float(glucose)
            if glucose < hypo_threshold: return 'Hypoglycemia'
            elif glucose <= hyper_threshold: return 'Euglycemia'
            return 'Hyperglycemia'
        except (ValueError, TypeError):
            return 'Invalid'

    states = glucose_series.apply(categorize_state)
    states = states[states != 'Invalid']
    fractions = (states.value_counts() / len(states)).to_dict()

    if plot:
        pd.Series(fractions).plot(kind='bar', color=['red', 'green', 'orange'])
        plt.title('Fraction of Time in Glycemic States')
        plt.xlabel('Glycemic State')
        plt.ylabel('Fraction of Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return fractions