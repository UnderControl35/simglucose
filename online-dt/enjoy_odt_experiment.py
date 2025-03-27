import gym
import numpy as np
import torch
import argparse
import os
from simglucose.envs import T1DSimEnv
from gym.envs.registration import register
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg

FILE_DIR = os.path.dirname(__file__)

def register_env(patient_name='adolescent#001'):
    env_id = f'simglucose-{patient_name.replace("#", "")}-v0'
    if env_id not in gym.envs.registry.env_specs:
        register(id=env_id, entry_point='simglucose.envs:T1DSimEnv', kwargs={'patient_name': patient_name})
    return env_id

def get_parser():
    parser = argparse.ArgumentParser(description="Decision Transformer Evaluation")
    parser.add_argument('--env', default='simglucose', type=str)
    parser.add_argument('--dataset', default='medium', choices=['medium', 'medium-replay', 'medium-expert', 'expert'])
    parser.add_argument('--mode', default='normal', choices=['normal', 'delayed'])
    parser.add_argument('--noisy', default=False, action='store_true', help="Use noisy (CGM) data")
    parser.add_argument('--model_type', default='dt', choices=['dt', 'bc'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_eval_episodes', default=100, type=int)
    parser.add_argument('--max_iters', default=10, type=int)
    parser.add_argument('--use_action_means', default=False, action='store_true')
    parser.add_argument('--stochastic_tanh', default=False, action='store_true')
    parser.add_argument('--approx_entropy_samples', default=1000, type=int)
    parser.add_argument('--eval_context', default=None, type=int)
    
    base_path = f'{FILE_DIR}/dataset/T1DatasetAnalysis'
    parser.add_argument('--algo', default='PPO', choices=['PPO', 'BB', 'PID'])
    parser.add_argument('--patient_name', default='adolescent#001', type=str)
    parser.add_argument('--noisy_path', default=f'{base_path}/{{algo}}/output_noisy/{{patient}}/{{patient}}_combined_seed.pkl', type=str)
    parser.add_argument('--normal_path', default=f'{base_path}/{{algo}}/output/{{patient}}/{{patient}}_combined_seed.pkl', type=str)
    parser.add_argument('--test', default=True, action='store_true')
    parser.add_argument('--savename', default='DT_{algo}_{patient}', type=str)
    return parser

def test_experiment(variant):
    device = variant['device']
    env_id = register_env(variant['patient_name'])
    env = gym.make(env_id)
    max_ep_len, scale = 480, 1000.
    env_targets = [180, 70]  # Evaluation targets
    state_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    load_dir = 'models/2025-03-15_20-11-51_online'
    state_mean = np.load(os.path.join(load_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(load_dir, "state_std.npy"))

    def evaluate_target(model, target_rew):
        returns, lengths = [], []
        for _ in range(variant['num_eval_episodes']):
            with torch.no_grad():
                if variant['model_type'] == 'dt':
                    ret, length = evaluate_episode_rtg(
                        env, state_dim, act_dim, model, max_ep_len=max_ep_len, scale=scale,
                        target_return=target_rew/scale, mode=variant['mode'], state_mean=state_mean,
                        state_std=state_std, device=device, use_means=variant['use_action_means'],
                        eval_context=variant['eval_context'], test=variant['test']
                    )
                else:
                    ret, length = evaluate_episode(
                        env, state_dim, act_dim, model, max_ep_len=max_ep_len,
                        target_return=target_rew/scale, mode=variant['mode'], state_mean=state_mean,
                        state_std=state_std, device=device
                    )
            returns.append(ret)
            lengths.append(length)
        return {
            f'target_{target_rew}_return_mean': np.mean(returns),
            f'target_{target_rew}_return_std': np.std(returns),
            f'target_{target_rew}_length_mean': np.mean(lengths),
            f'target_{target_rew}_length_std': np.std(lengths)
        }

    if variant['test']:
        model_path = os.path.join(load_dir, f"dt_gym-experiment-simglucose-{variant['dataset']}-864231.pt")
        model = torch.load(model_path, map_location=device)
        model.stochastic_tanh = variant['stochastic_tanh']
        model.approximate_entropy_samples = variant['approx_entropy_samples']
        model.to(device).eval()

        eval_fns = [lambda model: evaluate_target(model, tar) for tar in env_targets]
        for iter_num in range(variant['max_iters']):
            logs = {}
            for eval_fn in eval_fns:
                logs.update(eval_fn(model))
            print(f"{'=' * 80}\nIteration {iter_num}\n" + "\n".join(f"{k}: {v}" for k, v in logs.items()))

def main():
    args = get_parser().parse_args()
    args.datapath = (args.noisy_path if args.noisy else args.normal_path).format(algo=args.algo, patient=args.patient_name)
    test_experiment(vars(args))

if __name__ == '__main__':
    main()