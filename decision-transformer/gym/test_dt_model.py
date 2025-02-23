import gym
import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import sys
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from simglucose.envs import T1DSimEnv
from gym.envs.registration import register

# Register the simglucose environment
register(
    id='simglucose-adolescent1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#001'}
)

def discount_cumsum(x, gamma):
    """Compute discounted cumulative sum for returns-to-go."""
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def compute_glucose_metrics(bg_trajectory):
    """Compute glucose control metrics: euglycemia %, hypo/hyperglycemia %, and risk index."""
    # Define safe BG range (mg/dL) from the PLOS ONE paper
    euglycemic_range = (70, 180)  # Target range
    hypo_threshold = 70
    hyper_threshold = 350  # Conservative upper limit from paper

    # Convert trajectory to numpy if it's a tensor
    bg_trajectory = np.array(bg_trajectory)

    # Calculate time spent in each state
    euglycemic = np.sum((bg_trajectory >= euglycemic_range[0]) & (bg_trajectory <= euglycemic_range[1]))
    hypoglycemic = np.sum(bg_trajectory < hypo_threshold)
    hyperglycemic = np.sum(bg_trajectory > hyper_threshold)
    total_steps = len(bg_trajectory)

    euglycemic_pct = (euglycemic / total_steps) * 100
    hypoglycemic_pct = (hypoglycemic / total_steps) * 100
    hyperglycemic_pct = (hyperglycemic / total_steps) * 100

    # Clarke Blood Glucose Risk Index (BGRI) from paper
    def clarke_bgri(bg):
        f_bg = 1.509 * (np.log(bg) ** 1.084 - 5.381)
        r_bg = 10 * f_bg ** 2
        lbg = r_bg if f_bg < 0 else 0
        hbg = r_bg if f_bg > 0 else 0
        return lbg, hbg

    lbg_values, hbg_values = [], []
    for bg in bg_trajectory:
        lbg, hbg = clarke_bgri(bg)
        lbg_values.append(lbg)
        hbg_values.append(hbg)

    lbgi = np.mean(lbg_values)
    hbgi = np.mean(hbg_values)
    bgri = lbgi + hbgi

    return {
        'euglycemic_pct': euglycemic_pct,
        'hypoglycemic_pct': hypoglycemic_pct,
        'hyperglycemic_pct': hyperglycemic_pct,
        'lbgi': lbgi,
        'hbgi': hbgi,
        'bgri': bgri
    }

def test_model(variant, model_path=None, save_path=None):
    """Test the Decision Transformer model and optionally save it."""
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    # Environment setup
    env_name = variant['env']
    if env_name == 'simglucose':
        env = gym.make('simglucose-adolescent1-v0')
        max_ep_len = 480  # 1 day at 3-minute steps
        env_targets = [180, 70]  # Target BG levels for conditioning (mg/dL)
        scale = 1000.  # Normalization scale for returns
    else:
        raise NotImplementedError(f"Environment {env_name} not supported in test script.")

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load dataset for normalization (state mean/std)
    dataset_path = variant['dataset_path']
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    states = np.concatenate([traj['observations'] for traj in trajectories], axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    # Model setup
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant['K'],
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    model = model.to(device=device)

    # Load pre-trained model if provided
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print("No model path provided; using randomly initialized model (likely poor performance).")

    model.eval()

    # Evaluation
    num_eval_episodes = variant['num_eval_episodes']
    mode = variant.get('mode', 'delayed')
    results = {}
    all_bg_trajectories = []

    for target_rew in env_targets:
        returns, lengths, bg_trajectories = [], [], []
        for ep in range(num_eval_episodes):
            with torch.no_grad():
                ret, length, bg_traj = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    return_bg_trajectory=True  # Custom modification to get BG trajectory
                )
            returns.append(ret)
            lengths.append(length)
            bg_trajectories.append(bg_traj)

        # Compute standard metrics
        results[f'target_{target_rew}_return_mean'] = np.mean(returns)
        results[f'target_{target_rew}_return_std'] = np.std(returns)
        results[f'target_{target_rew}_length_mean'] = np.mean(lengths)
        results[f'target_{target_rew}_length_std'] = np.std(lengths)

        # Compute glucose-specific metrics
        metrics = [compute_glucose_metrics(traj) for traj in bg_trajectories]
        results[f'target_{target_rew}_euglycemic_pct'] = np.mean([m['euglycemic_pct'] for m in metrics])
        results[f'target_{target_rew}_hypoglycemic_pct'] = np.mean([m['hypoglycemic_pct'] for m in metrics])
        results[f'target_{target_rew}_hyperglycemic_pct'] = np.mean([m['hyperglycemic_pct'] for m in metrics])
        results[f'target_{target_rew}_bgri'] = np.mean([m['bgri'] for m in metrics])

        all_bg_trajectories.extend(bg_trajectories)

    # Print results
    print('=' * 50)
    print(f"Evaluation results for {env_name}:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")
    print('=' * 50)

    # Save model if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Log to Weights & Biases if enabled
    if log_to_wandb:
        wandb.init(project='decision-transformer-test', config=variant)
        wandb.log(results)
        # Optionally log a sample BG trajectory
        wandb.log({"bg_trajectory_sample": wandb.plot.line_series(
            xs=[np.arange(len(all_bg_trajectories[0])) * 3 / 60],  # Convert steps to hours
            ys=[all_bg_trajectories[0]],
            keys=["BG (mg/dL)"],
            title="Sample BG Trajectory",
            xname="Time (hours)"
        )})

    return results, all_bg_trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='simglucose', help="Environment name")
    parser.add_argument('--dataset_path', type=str, 
                        default='/home/guleserhocam/VS_Projects/simglucose/dataset/T1DatasetAnalysis/BB/output/adolescent#001/adolescent#001_combined_seed.pkl',
                        help="Path to preprocessed dataset for normalization")
    parser.add_argument('--mode', type=str, default='delayed', help="Evaluation mode: normal or delayed")
    parser.add_argument('--K', type=int, default=20, help="Context length for DT")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (unused in testing)")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--n_layer', type=int, default=3, help="Number of transformer layers")
    parser.add_argument('--n_head', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--activation_function', type=str, default='relu', help="Activation function")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--num_eval_episodes', type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on")
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False, help="Log to Weights & Biases")
    parser.add_argument('--model_path', type=str, default=None, help="Path to load pre-trained model")
    parser.add_argument('--save_path', type=str, default='./models/dt_simglucose.pth', help="Path to save model")

    args = parser.parse_args()
    variant = vars(args)

    # Run the test
    results, bg_trajectories = test_model(variant, model_path=args.model_path, save_path=args.save_path)