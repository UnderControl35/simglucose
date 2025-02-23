import os, sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR)

def load_pickle_file(file_path):
    """
    Loads trajectory data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        list: List of trajectory dictionaries.
    """
    if not os.path.isfile(file_path):
        print(f"Pickle file not found: {file_path}")
        return None

    with open(file_path, 'rb') as f:
        trajectories = pkl.load(f)

    print(f"Loaded {len(trajectories)} trajectories from {file_path}")
    return trajectories

def analyze_trajectories(trajectories):
    """
    Analyzes trajectories and computes basic statistics.

    Args:
        trajectories (list): List of trajectory dictionaries.

    Returns:
        pd.DataFrame: Combined DataFrame with all data points.
    """
    combined_data = {
        "Trajectory": [],
        "Step": [],
        "Blood Glucose (BG)": [],
        "Reward": []
    }

    # Combine all trajectories
    for i, traj in enumerate(trajectories):
        steps = range(len(traj['observations']))
        glucose = [obs[0] for obs in traj['observations']]
        rewards = traj['rewards']

        combined_data["Trajectory"].extend([i] * len(steps))
        combined_data["Step"].extend(steps)
        combined_data["Blood Glucose (BG)"].extend(glucose)
        combined_data["Reward"].extend(rewards)

    # Convert to DataFrame
    df = pd.DataFrame(combined_data)

    # Print basic statistics
    print("\nStatistical Summary:")
    print(df.describe())

    return df

def visualize_data(df, output_folder):
    """
    Visualizes blood glucose and reward data.

    Args:
        df (pd.DataFrame): DataFrame containing trajectory data.
        output_folder (str): Folder to save visualizations.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Plot Blood Glucose over Steps
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Step", y="Blood Glucose (BG)", hue="Trajectory", palette="tab10", legend=False)
    plt.axhline(70, color="red", linestyle="--", label="Low Threshold (70 mg/dL)")
    plt.axhline(180, color="orange", linestyle="--", label="High Threshold (180 mg/dL)")
    plt.title("Blood Glucose Levels Over Time", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Blood Glucose (mg/dL)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "blood_glucose_over_time.png"))
    plt.show()

    # Plot Reward Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df["Reward"], bins=20, kde=True, color="green")
    plt.title("Reward Distribution", fontsize=16)
    plt.xlabel("Reward", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "reward_distribution.png"))
    plt.show()

def main():
    # Specify the pickle file path and output folder
    folder = 'BB/output'
    patient_name = 'adolescent#001'
    pickle_file = os.path.join(FILE_DIR, folder, patient_name, f"{patient_name}_combined_seed.pkl")
    output_folder = os.path.join(FILE_DIR, folder, patient_name, "analysis")

    # Load the pickle file
    trajectories = load_pickle_file(pickle_file)
    if trajectories is None:
        return

    # Analyze trajectories
    df = analyze_trajectories(trajectories)

    # Visualize data
    visualize_data(df, output_folder)

if __name__ == "__main__":
    main()
