import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import argparse

FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR)

SAVE_DIR = 'output'  # Change this to your desired output directory

def load_patient_data(folder_path, patients, numbers, seeds, file_suffix=""):
    """
    Loads and combines patient data from multiple CSV files across folder structures.

    Args:
        folder_path (str): Base folder path containing subfolders or files.
        patient_name (str): Patient identifier (e.g., 'adolescent#001').
        seeds (list): List of seed subfolder names or suffixes to check.
        file_suffix (str): Optional suffix for file names.

    Returns:
        list: Trajectory data for the patient.
    """
    for patient in patients:
        for number in numbers:
            patient_name = f"{patient}{number}"
            traj = []  # To store all trajectory data
            for seed in seeds:
                # Construct file path
                filename = f"{patient_name}{file_suffix}.csv"
                if seed:
                    file_path = os.path.join(FILE_DIR, folder_path, seed, filename)
                else:
                    file_path = os.path.join(FILE_DIR, folder_path, filename)

                if os.path.isfile(file_path):
                    # Read the CSV file
                    df = pd.read_csv(file_path).copy()
                    traj_len = len(df['BG'])

                    # Add 'done' column
                    df['done'] = np.array([False] * traj_len, dtype=bool)
                    df.loc[traj_len - 1, 'done'] = True # Mark the last step as done

                    # Extract observations and actions
                    obs = [np.array([bg], dtype=np.float32) for bg in df['BG']]
                    if folder_path=='BB' or folder_path=='PID':
                        # Extract next observations (shifted BG, last one repeats or uses a convention)
                        next_obs = np.roll(obs, -1, axis=0)  # Shift observations up, last one wraps around
                        next_obs[-1] = obs[-1]  # Convention: last next_obs is same as last obs (terminal) 
                        acts = [np.array([action], dtype=np.float32) for action in df['insulin']]
                        # Replace NaN in actions with 0, preserving shape
                        acts = np.where(np.isnan(acts), 0.0, acts).astype(np.float32)
                        # Ensure actions has the same shape as next_observations
                        assert acts.shape == next_obs.shape, f"Shape mismatch: actions {acts.shape} vs next_observations {next_obs.shape}"
                        rewards = [np.array(create_reward(bg), dtype=np.float32) for bg in df['BG']] 
                        terminals = df['done'].values
                    else:
                        # Extract next observations (shifted BG, last one repeats or uses a convention)
                        next_obs = np.roll(obs, -1, axis=0)  # Shift observations up, last one wraps around
                        next_obs[-1] = obs[-1]  # Convention: last next_obs is same as last obs (terminal)  
                        acts = [np.array([action], dtype=np.float32) for action in df['Action']]
                        rewards = df['reward'].values
                        terminals = df['done'].values

                    # Append data to trajectory
                    traj.append({
                        'observations': np.asarray(obs, dtype=np.float32),
                        'next_observations': np.asarray(next_obs, dtype=np.float32),
                        'actions': np.asarray(acts, dtype=np.float32),
                        'rewards': np.asarray(rewards, dtype=np.float32),
                        'terminals': np.asarray(terminals, dtype=bool)
                    })
                else:
                    print(f"File not found: {file_path}")

            print(f"Loaded {len(traj)} trajectories for {patient_name}.")
            # Save as pickle
            save_patient_data_as_pickle(traj, SAVE_DIR, patient_name, folder_path)
    return traj

def save_patient_data_as_pickle(trajectories, base_output_path, patient_name, folder_path):
    """
    Saves combined patient data as a pickle file in a folder named after the patient.

    Args:
        trajectories (list): Combined trajectory data for the patient.
        base_output_path (str): Base directory where the pickle file will be saved.
        patient_name (str): Patient identifier (used as folder name).
    """
    file_path = os.path.join(FILE_DIR, folder_path)
    # Create the patient's folder if it doesn't exist
    patient_folder = os.path.join(file_path, base_output_path, patient_name)
    os.makedirs(patient_folder, exist_ok=True)

    # Define the output pickle file path
    output_file = os.path.join(patient_folder, f"{patient_name}_combined_seed.pkl")

    # Save data to pickle file
    with open(output_file, 'wb') as f:
        pkl.dump(trajectories, f)

    print(f"Data saved successfully to {output_file}")

def create_reward(glucose_level):
    if glucose_level > 180:
        return -1
    elif glucose_level < 70:
        return -2
    else:
        return 1
    

def plot_single_trajectory(folder, patient_name, base_output_path, trajectory_index=0):
    """
    Reads a patient's trajectory from a pickle file and plots a single trajectory
    with both Blood Glucose and Reward values on the same figure.
    
    Also displays the mean and standard deviation of Blood Glucose levels.
    """
    # Locate the pickle file in the patient's folder
    pickle_path = os.path.join(FILE_DIR, folder, base_output_path, patient_name, f"{patient_name}_combined_seed.pkl")
    
    if not os.path.isfile(pickle_path):
        print(f"Pickle file not found: {pickle_path}")
        return

    # Load the pickle file
    with open(pickle_path, 'rb') as f:
        trajectories = pkl.load(f)

    # Validate trajectory index
    if trajectory_index < 0 or trajectory_index >= len(trajectories):
        print(f"Invalid trajectory index: {trajectory_index}. Total trajectories: {len(trajectories)}")
        return

    # Extract the specified trajectory
    traj = trajectories[trajectory_index]
    steps = range(len(traj['observations']))
    glucose = [obs[0] for obs in traj['observations']]
    rewards = traj['rewards']

    # Compute statistics
    mean_glucose = np.mean(glucose)
    std_glucose = np.std(glucose)

    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)

    # Create a DataFrame for the trajectory
    df = pd.DataFrame({
        "Step": steps,
        "Blood Glucose (BG)": glucose,
        "Reward": rewards
    })

    # Seaborn settings
    sns.set(style="whitegrid", context="paper")
    fig, ax1 = plt.subplots(figsize=(16, 10))

    # Plot Blood Glucose on the primary y-axis
    sns.lineplot(data=df, x="Step", y="Blood Glucose (BG)", ax=ax1, color="blue", label=f"Blood Glucose (Mean: {mean_glucose:.2f}, Std: {std_glucose:.2f})")
    ax1.axhline(70, color="red", linestyle="--", label="Low Threshold (70 mg/dL)")
    ax1.axhline(180, color="orange", linestyle="--", label="High Threshold (180 mg/dL)")

    # Primary y-axis settings
    ax1.set_xlabel("Step", fontsize=14)
    ax1.set_ylabel("Blood Glucose (mg/dL)", fontsize=14, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="lower left", fontsize=12)

    # Create a secondary y-axis for rewards
    ax2 = ax1.twinx()
    sns.scatterplot(data=df, x="Step", y="Reward", ax=ax2, color="green", linestyle="--", \
                    label="Reward (Mean: {:.2f}, Std: {:.2f}, Total: {:.2f})".format(mean_rewards, std_rewards, sum(rewards)))

    # Secondary y-axis settings
    ax2.set_ylabel("Reward", fontsize=14, color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.legend(loc="lower right", fontsize=12)

    # Title and layout
    plt.title(f"Blood Glucose and Reward for {patient_name} (Trajectory {trajectory_index})", fontsize=18)
    plt.tight_layout()

    # Show the plot
    plt.show()




if __name__ == "__main__":


    # Argument parser setup
    parser = argparse.ArgumentParser(description="Plot a single trajectory from patient data.")
    parser.add_argument("--folder", type=str, required=False, help="Folder name (e.g., PPO, BB, PID).")
    parser.add_argument("--patient_name", type=str, required=False, help="Patient name (e.g., adolescent#001).")
    parser.add_argument("--trajectory_index", type=int, default=0, help="Trajectory index to plot (default: 0).")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Directory where output is saved (default: 'output').")

    # Parse arguments
    args = parser.parse_args()

    # Load data
    trajectories = load_patient_data(folder_path='PPO', 
                                     patients=['adolescent', 'child', 'adult'], 
                                     numbers=[f'#{i:03d}' for i in range(1, 11)], 
                                     seeds=[f'seed{i}' for i in range(20)])
    
    # Load data
    trajectories = load_patient_data(folder_path='BB', 
                                     patients=['adolescent', 'child', 'adult'], 
                                     numbers=[f'#{i:03d}' for i in range(1, 11)], 
                                     seeds=[f'results{i}' for i in range(20)])
    
    # Load data
    trajectories = load_patient_data(folder_path='PID', 
                                     patients=['adolescent', 'child', 'adult'], 
                                     numbers=[f'#{i:03d}' for i in range(1, 11)], 
                                     seeds=[f'results{i}' for i in range(20)])
    
    # Call the function with parsed arguments
    # plot_single_trajectory(
    #     folder=args.folder,
    #     patient_name=args.patient_name,
    #     base_output_path=args.save_dir,
    #     trajectory_index=args.trajectory_index
    # )



