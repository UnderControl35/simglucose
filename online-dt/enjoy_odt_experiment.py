import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os
import pathlib
from datetime import datetime

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

import simglucose
from gym.envs.registration import register

register(
    id='simglucose-adolescent1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#001'}
)

def test_experiment(variant):

    device = variant.get('device', 'cuda')
    
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']

    num_eval_episodes = variant['num_eval_episodes']

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []

    load_dir = 'models/2025-03-15_20-11-51_online'
    # If you saved these during training, load them instead
    state_mean = np.load(os.path.join(load_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(load_dir, "state_std.npy"))
    

    if env_name == 'simglucose':
        env = gym.make('simglucose-adolescent1-v0')
        max_ep_len = 480
        env_targets = [180, 70]
        scale = 1000.
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError
    
    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_means=variant['use_action_means'],
                            eval_context=variant['eval_context'],
                            test=variant['test'],
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn
    
    if variant['test']:
            # Define model architecture (must match your trained model)
            model = torch.load('models/2025-03-15_20-11-51_online/dt_gym-experiment-simglucose-medium-864231.pt',map_location = device)
            model.stochastic_tanh = variant['stochastic_tanh']
            model.approximate_entropy_samples = variant['approximate_entropy_samples']
            model.to(device)

            model.eval()
            eval_fns = [eval_episodes(tar) for tar in env_targets]
            
            for iter_num in range(variant['max_iters']):
                logs = {}
                for eval_fn in eval_fns:
                    outputs = eval_fn(model)
                    for k, v in outputs.items():
                        logs[f'evaluation/{k}'] = v

                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    #OfflineParams
    parser.add_argument('--env', type=str, default='simglucose')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)

    parser.add_argument('--use_action_means', default=False, action='store_true')
    parser.add_argument('--stochastic_tanh', default=False, action='store_true')
    parser.add_argument('--approximate_entropy_samples',default=1000, type=int, 
                        help="if using stochastic network w/ tanh squashing, have to approximate entropy with k samples, as no anlytical solution")
    parser.add_argument('--eval_context', default=None, type=int)
    
    default_patient = 'adolescent#001'
    algo = 'PPO'
    default_path = f'/home/guleserhocam/VS_Projects/simglucose/dataset/T1DatasetAnalysis/{algo}/output/{default_patient}/{default_patient}_combined_seed.pkl'

    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--patient_name', type=str, default=f'{default_patient}')
    parser.add_argument('--algo', type=str, default=f'{algo}')
    parser.add_argument('--savename', type=str, default=f'DT_{algo}_{default_patient}')
    args = parser.parse_args()

    test_experiment(variant=vars(args))