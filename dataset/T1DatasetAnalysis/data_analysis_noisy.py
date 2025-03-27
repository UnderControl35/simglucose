import argparse
import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_DIR = os.path.dirname(__file__)

def load_trajectory(file_path, idx=0):
    """Load a trajectory from a pickle file or return None if invalid."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None
    with open(file_path, 'rb') as f:
        trajs = pkl.load(f)
    if idx < 0 or idx >= len(trajs):
        print(f"Invalid index {idx}. Total: {len(trajs)}")
        return None
    print(f"Loaded trajectory {idx} from {file_path} (Total: {len(trajs)})")
    return trajs[idx]

def analyze_trajectory(noisy, gt):
    """Create DataFrame from noisy (CGM) and ground truth (BG) trajectories."""
    df = pd.DataFrame({
        "Step": range(len(noisy['observations'])),
        "CGM": [obs[0] for obs in noisy['observations']],
        "BG": [obs[0] for obs in gt['observations']],
        "Reward": noisy['rewards']
    })
    print("\nStats:\n", df.describe())
    return df

def plot_trajectory(df, method, patient, idx, out_dir):
    """Plot CGM, BG, and rewards with dual axes."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Glucose plots
    sns.lineplot(data=df, x="Step", y="CGM", ax=ax1, color="blue", label="Noisy (CGM)", ls="--")
    sns.lineplot(data=df, x="Step", y="BG", ax=ax1, color="red", label="Ground Truth (BG)", ls="-")
    ax1.axhline(70, c="purple", ls="--", label="Low (70 mg/dL)")
    ax1.axhline(180, c="orange", ls="--", label="High (180 mg/dL)")
    ax1.set_ylabel("Glucose (mg/dL)", c="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.legend(loc="lower left")

    # Reward plot
    ax2 = ax1.twinx()
    sns.scatterplot(data=df, x="Step", y="Reward", ax=ax2, color="green", label="Reward")
    ax2.set_ylabel("Reward", c="green")
    ax2.tick_params(axis='y', labelcolor="green")
    ax2.legend(loc="lower right")

    plt.title(f"{method} Trajectory {idx} - {patient}", fontsize=16)
    ax1.set_xlabel("Step", fontsize=14)
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"{method}_traj_{idx}.png")
    plt.savefig(out_file)
    print(f"Saved to {out_file}")
    #plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot trajectories.")
    parser.add_argument("--method", default="BB", choices=["BB", "PPO", "PID"], help="Method (BB, PPO, PID)")
    parser.add_argument("--patient", default="adolescent#001", help="Patient name")
    parser.add_argument("--index", type=int, default=0, help="Trajectory index")
    args = parser.parse_args()

    # Paths
    noisy_path = os.path.join(FILE_DIR, f"{args.method}/output_noisy", args.patient, f"{args.patient}_combined_seed.pkl")
    gt_path = os.path.join(FILE_DIR, f"{args.method}/output", args.patient, f"{args.patient}_combined_seed.pkl")
    out_dir = os.path.join(FILE_DIR, f"{args.method}/output_noisy", args.patient, "analysis")

    # Load and process
    noisy = load_trajectory(noisy_path, args.index)
    gt = load_trajectory(gt_path, args.index)
    if noisy and gt:
        df = analyze_trajectory(noisy, gt)
        plot_trajectory(df, args.method, args.patient, args.index, out_dir)

if __name__ == "__main__":
    main()