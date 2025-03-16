import gym
import numpy as np
import torch
import argparse
import pickle
import random
import sys
import os
import pathlib
from datetime import datetime
import wandb

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel

import simglucose
from gym.envs.registration import register

def convert_patient_id(patient_id):
    prefix, number = patient_id.split("#")
    number = str(int(number))
    return prefix + number

# Register the environment once at the module level
default_patient = 'adolescent#001'
patient_id = convert_patient_id(default_patient)
env_id = f'simglucose-{patient_id}-v0'

# Check if already registered to avoid re-registration
if env_id not in gym.envs.registry.env_specs:
    register(
        id=env_id,
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': default_patient}
    )

def test_decision_transformer(variants):
    # Extract variables from variants with defaults for swept parameters
    device = variants['device']
    env_name = variants['env']
    patient_name = variants['patient_name']
    algo = variants.get('algo', 'BB').upper()
    model_dir = variants['model_dir']
    model_type = variants['model_type']
    num_episodes = variants.get('num_eval_episodes', 1)
    max_iters = variants['max_iters']
    debug = variants.get('debug', False)

    # Define algorithm-specific paths
    algo_paths = {
        'BB': '/home/guleserhocam/VS_Projects/simglucose/models/2025-02-26_06-39-37_offline',
        'PID': '/home/guleserhocam/VS_Projects/simglucose/models/2025-02-27_06-23-42_offline',
        'PPO': '/home/guleserhocam/VS_Projects/simglucose/models/2025-02-27_14-07-22_offline'
    }

    if algo not in algo_paths:
        raise ValueError(f"Unsupported algorithm: {algo}. Supported: {list(algo_paths.keys())}")
    
    model_path = algo_paths[algo]

    # Environment setup (no registration here, already done at module level)
    if env_name == 'simglucose':
        env = gym.make(env_id)  # Use pre-registered ID
        max_ep_len = 480
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        env_targets = variants.get('env_targets', [4800])
        scale = 1000.
    else:
        raise NotImplementedError(f"Environment {env_name} not supported for this sweep")

    K = variants.get('K', 20)

    # Define model architecture
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=128,
        n_layer=3,
        n_head=4,
        n_inner=4 * 128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1
    )

    model_path_file = os.path.join(model_path, f"DT_{algo}_{patient_name}.pth")
    if not os.path.exists(model_path_file):
        raise FileNotFoundError(f"Model file not found: {model_path_file}")

    state_dict = torch.load(model_path_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    state_mean = np.load(os.path.join(model_path, "state_mean.npy"))
    state_std = np.load(os.path.join(model_path, "state_std.npy"))

    def eval_episode(target_return):
        returns = []
        lengths = []
        
        for episode in range(num_episodes):
            with torch.no_grad():
                ret, length = evaluate_episode_rtg(
                    env=env,
                    state_dim=state_dim,
                    act_dim=act_dim,
                    model=model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_return/scale,
                    mode='normal',
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    render=variants['render'],
                    debug=debug
                )
            returns.append(ret)
            lengths.append(length)
            if debug:
                print(f"Episode {episode + 1}: Return = {ret:.2f}, Length = {length}")
        
        return returns, lengths

    # Initialize W&B run
    with wandb.init(config=variants):
        # Update variants with sweep config
        variants.update(wandb.config)

        print(f"\nTesting {algo} model on {env_name} environment")
        print("=" * 50)
        
        for iter_num in range(max_iters):
            print(f"\nIteration {iter_num + 1}")
            for target in env_targets:
                print(f"\nEvaluating with target return: {target}")
                returns, lengths = eval_episode(target)
                
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                mean_length = np.mean(lengths)
                success_rate = np.mean(np.array(returns) >= target) * 100
                
                # Log metrics to W&B
                wandb.log({
                    f"{algo}/target_{target}_mean_return": mean_return,
                    f"{algo}/target_{target}_std_return": std_return,
                    f"{algo}/target_{target}_mean_length": mean_length,
                    f"{algo}/target_{target}_success_rate": success_rate,
                    "iteration": iter_num + 1
                })

                print(f"\nResults for target {target}:")
                print(f"Average Return: {mean_return:.3f} ± {std_return:.3f}")
                print(f"Average Episode Length: {mean_length:.3f}")
                print(f"Success Rate (>= Target): {success_rate:.3f}%")

def run_sweep():
    # Define sweep configuration for SimGlucose evaluation
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'DT/target_4800_mean_return',  # Optimize for BB's highest target
            'goal': 'maximize'
        },
        'parameters': {
            'env_targets': {
                'values': [[1800], [3600], [4800]]  # SimGlucose-specific targets (e.g., minutes or BG-related)
            },
            'K': {
                'values': [10, 20, 30]
            },
            'algo': {
                'values': ['BB', 'PID', 'PPO']
            }
        }
    }

    # Base configuration (fixed parameters with defaults for swept keys)
    base_config = {
        'render': False,
        'env': 'simglucose',
        'model_type': 'dt',
        'device': 'cuda',
        'max_iters': 1,
        'debug': False,
        'model_dir': None,
        'patient_name': default_patient,
        'algo': 'BB',
        'num_eval_episodes': 1,
        'env_targets': [4800],  # Default to max SimGlucose episode length as target
        'K': 20
    }

    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="simglucose-eval-sweep")

    # Run sweep
    wandb.agent(sweep_id, lambda: test_decision_transformer(base_config))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--sweep', action='store_true', help="Run evaluation-only hyperparameter sweep")
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--env', type=str, default='simglucose')
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=1)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model_dir', type=str, default='/home/guleserhocam/VS_Projects/simglucose/models/2025-02-26_06-39-37_offline')
    parser.add_argument('--patient_name', type=str, default=default_patient)
    parser.add_argument('--algo', type=str, default='BB', choices=['BB', 'PID', 'PPO'], help="Algorithm: BB, PID or PPO")

    args = parser.parse_args()

    if args.sweep:
        run_sweep()
    else:
        variants = vars(args)
        algos_to_test = [args.algo] if args.algo in ['BB', 'PID', 'PPO'] else ['BB', 'PID', 'PPO']
        for algo in algos_to_test:
            variants['algo'] = algo
            test_decision_transformer(variants)