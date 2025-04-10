import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

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

def analyze_glycemic_states_violin(glucose_data, interval_minutes=60, time_step=5, plot=False, 
                                 hypo_threshold=70, hyper_threshold=180):
    """
    Analyze glycemic data and create a violin plot of glucose values by glycemic state.
    
    Parameters:
    - glucose_data (list): List of glucose readings (assumed to be evenly spaced).
    - interval_minutes (int): Time interval for binning (e.g., 60 for hourly analysis).
    - time_step (int): Time between consecutive readings in minutes (default: 5).
    - plot (bool): If True, display a violin plot using Seaborn.
    - hypo_threshold (float): Hypoglycemia threshold (default: 70 mg/dL).
    - hyper_threshold (float): Hyperglycemia threshold (default: 180 mg/dL).
    
    Returns:
    - dict: Contains glucose data with states and overall glycemic fractions.
    """
    # Convert glucose data to a Pandas Series
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
    valid_mask = states != 'Invalid'
    glucose_series = glucose_series[valid_mask]
    states = states[valid_mask]

    # Check if there's any valid data after filtering
    if len(glucose_series) == 0:
        return {
            'glucose_data': pd.DataFrame(columns=['Glucose', 'State']),
            'interval_fractions': pd.DataFrame(),
            'overall_fractions': {}
        }

    # Create time index and DataFrame
    num_readings = len(glucose_series)
    timestamps = pd.date_range(start='2025-01-01', periods=num_readings, freq=f'{time_step}min')
    
    # Ensure glucose_series and states are aligned with timestamps
    df = pd.DataFrame({
        'Glucose': glucose_series.values,  # Use .values to avoid index mismatch
        'State': states.values
    }, index=timestamps)

    # Resample into intervals and compute fractions
    interval_freq = f'{interval_minutes}min'
    resampled = df.resample(interval_freq)
    
    # Compute state counts and fractions, handling empty intervals
    state_counts = resampled['State'].value_counts().unstack(fill_value=0)
    total_counts = state_counts.sum(axis=1)
    fractions = state_counts.div(total_counts, axis=0).fillna(0)

    # Overall fractions
    overall_fractions = (states.value_counts() / len(states)).to_dict()

    # Result dictionary
    result = {
        'glucose_data': df,
        'interval_fractions': fractions,
        'overall_fractions': overall_fractions
    }

    # Create violin plot with Seaborn
    if plot:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        sns.violinplot(x='State', y='Glucose', data=df,
                      order=['Hypoglycemia', 'Euglycemia', 'Hyperglycemia'],
                      palette='muted',
                      inner='quartile')

        plt.axhline(y=hypo_threshold, color='red', linestyle='--', 
                   label=f'Hypoglycemia Threshold ({hypo_threshold} mg/dL)')
        plt.axhline(y=hyper_threshold, color='orange', linestyle='--', 
                   label=f'Hyperglycemia Threshold ({hyper_threshold} mg/dL)')

        plt.xlabel('Glycemic State')
        plt.ylabel('Glucose (mg/dL)')
        plt.title('Distribution of Glucose Values by Glycemic State')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result