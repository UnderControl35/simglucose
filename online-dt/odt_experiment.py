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


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    
    
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    # model_dir = os.path.join(pathlib.Path(__file__).parent.resolve(),f'./models/{env_name}/')
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    if env_name == 'simglucose':
        env = gym.make('simglucose-adolescent1-v0')
        max_ep_len = 480
        env_targets = [180, 70]
        scale = 1000.
    elif env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError
    
    # Override env_targets / set different training target for online decision transformer, following paper
    if variant['online_training']:
        if env_name == 'simglucose':
            env_targets = [180]
            target_online = 360

        elif env_name == 'hopper':
            env_targets = [3600]  # evaluation conditioning targets
            target_online = 7200
        elif env_name == 'halfcheetah':
            env_targets = [6000]
            target_online = 12000
        elif env_name == 'walker2d':
            env_targets = [5000]
            target_online = 10000
        else:
            raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = variant['datapath']
    #dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # Save state_mean and state_std as .npy files
    if variant['online_training']:
        model_dir = f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_online"
    else:
        model_dir = f"models/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_offline"  # Create a directory for this experiment
    os.makedirs(model_dir, exist_ok=True)    # Ensure directory exists
    np.save(os.path.join(model_dir, "state_mean.npy"), state_mean)
    np.save(os.path.join(model_dir, "state_std.npy"), state_std)
    print(f"Saved state_mean and state_std to {model_dir}")
    
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    # Sort trajectories from worst to best and cut to buffer size
    if variant['online_training']:
        trajectories = [trajectories[index] for index in sorted_inds]
        trajectories = trajectories[:variant['online_buffer_size']]
        num_trajectories = len(trajectories)

    starting_p_sample = p_sample
    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training
        if variant['online_training']:
            traj_lens = np.array([len(path['observations']) for path in trajectories])
            p_sample = traj_lens / sum(traj_lens)
        else:
            p_sample = starting_p_sample
            
        
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            if variant['online_training']:
                traj = trajectories[batch_inds[i]]
            else:
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * 0., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    if variant['online_training']:
        # If online training, use means during eval, but (not during exploration)
        variant['use_action_means'] = True
    
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


    if model_type == 'dt':
        #FIXME: Save whole model in next trainings
        if variant['pretrained_model']:
            # Define model architecture (must match your trained model)
            model = torch.load(variant['pretrained_model'],map_location='cuda:0')
            model.stochastic_tanh = variant['stochastic_tanh']
            model.approximate_entropy_samples = variant['approximate_entropy_samples']
            model.to(device)

        else:
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len*2,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                stochastic = variant['stochastic'],
                remove_pos_embs=variant['remove_pos_embs'],
                approximate_entropy_samples = variant['approximate_entropy_samples'],
                stochastic_tanh=variant['stochastic_tanh']
            )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
        
    if variant['online_training']:
        assert(variant['pretrained_model'] is not None), "Must specify pretrained model to perform online finetuning"
        variant['use_entropy'] = True
        
    if variant['online_training'] and variant['target_entropy']:
        # Setup variable and optimizer for (log of) lagrangian multiplier used for entropy constraint
        # We optimize the log of the multiplier b/c lambda >= 0
        log_entropy_multiplier = torch.zeros(1, requires_grad=True, device=device)
        multiplier_optimizer = torch.optim.AdamW(
            [log_entropy_multiplier],
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        # multiplier_optimizer = torch.optim.Adam(
        #     [log_entropy_multiplier],
        #     lr=1e-3
        #     #lr=variant['learning_rate'],
        # )
        multiplier_scheduler = torch.optim.lr_scheduler.LambdaLR(
            multiplier_optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
    else:
        log_entropy_multiplier = None
        multiplier_optimizer = None 
        multiplier_scheduler = None

    entropy_loss_fn = None
    if variant['stochastic']:
        if variant['use_entropy']:
            if variant['target_entropy']:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.exp(log_entropy_multiplier.detach()) * torch.mean(entropies)
                target_entropy = -act_dim
                entropy_loss_fn = lambda entropies: torch.exp(log_entropy_multiplier) * (torch.mean(entropies.detach()) - target_entropy)
            else:
                loss_fn = lambda s_hat, a_hat, rtg_hat,r_hat, s, a, rtg, r, a_log_prob, entropies: -torch.mean(a_log_prob) - torch.mean(entropies)
        else:
            loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg,r, a_log_prob, entropies: -torch.mean(a_log_prob)
    else:
        loss_fn = lambda s_hat, a_hat, rtg_hat, r_hat, s, a, rtg, r, a_log_prob, entropies: torch.mean((a_hat - a)**2)

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            log_entropy_multiplier=log_entropy_multiplier,
            entropy_loss_fn=entropy_loss_fn,
            multiplier_optimizer=multiplier_optimizer,
            multiplier_scheduler=multiplier_scheduler,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=loss_fn,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
    if variant['eval_only']:
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
    else:
        if variant['online_training']:
            for iter in range(variant['max_iters']):
                # Collect new rollout, using stochastic policy
                ret, length, traj = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_online/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            use_means=False,
                            return_traj=True,
                            test=variant['test'],
                )
                # Remove oldest trajectory, add new trajectory
                trajectories = trajectories[1:]
                trajectories.append(traj)
                
                # Perform update, eval using deterministic policy 
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)
        else:
            for iter in range(variant['max_iters']):
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)

        torch.save(model,os.path.join(model_dir, model_type + '_' + exp_prefix + '.pt'))


if __name__ == '__main__':
    # Reasoning for '--env': 'simglucose' chosen as the target environment to simulate Type 1 diabetes management,
    # aligning with the goal of optimizing blood glucose (BG) control for virtual patients.

    # Reasoning for '--dataset': 'medium' selected as a balanced offline dataset for pretraining,
    # providing reasonable BG control trajectories without requiring expert-level data, suitable for SimGlucose adaptation.

    # Reasoning for '--mode': 'normal' used as the standard setting for dense rewards in SimGlucose,
    # avoiding sparse 'delayed' mode since BG feedback is continuous, simplifying training.

    # Reasoning for '--K': 20 timesteps (~100 min with 5-min steps) captures short-term BG dynamics in SimGlucose,
    # balancing context length with computational efficiency per the transformer’s design.

    # Reasoning for '--pct_traj': 1.0 uses all available trajectories from the dataset,
    # maximizing data for SimGlucose’s patient-specific offline pretraining.

    # Reasoning for '--batch_size': 64 aligns with typical transformer batch sizes,
    # balancing memory usage and gradient stability for SimGlucose’s state-action pairs.

    # Reasoning for '--model_type': 'dt' chosen for Decision Transformer’s ability to model sequential decisions,
    # ideal for SimGlucose’s time-series BG control over Behavior Cloning’s simpler imitation.

    # Reasoning for '--embed_dim': 128 provides sufficient capacity for embedding SimGlucose’s state and action spaces,
    # reduced from larger values to optimize compute for a smaller problem scope.

    # Reasoning for '--n_layer': 3 layers match the original DT paper’s shallow architecture,
    # sufficient for SimGlucose’s relatively simple dynamics while keeping training fast.

    # Reasoning for '--n_head': 1 head simplifies attention, adequate for SimGlucose’s focused state dependencies,
    # reducing complexity from multi-head setups in larger domains.

    # Reasoning for '--activation_function': 'relu' chosen for its simplicity and effectiveness in transformers,
    # standard for DT and suitable for SimGlucose’s continuous BG predictions.

    # Reasoning for '--dropout': 0.1 aligns with the DT paper’s regularization,
    # preventing overfitting to SimGlucose’s offline trajectories while maintaining generalization.

    # Reasoning for '--learning_rate' ('-lr'): 1e-4 follows the DT paper’s conservative rate,
    # ensuring stable convergence for SimGlucose’s sensitive insulin dosing adjustments.

    # Reasoning for '--weight_decay' ('-wd'): 1e-4 adds light L2 regularization per the DT paper,
    # controlling model complexity for SimGlucose’s limited patient variability.

    # Reasoning for '--warmup_steps': 1000 steps provide a gradual LR increase per the DT paper,
    # stabilizing early training for SimGlucose’s offline phase with fewer steps than larger tasks.

    # Reasoning for '--num_eval_episodes': 100 episodes allow robust evaluation of BG stability (euglycemia %, risk index),
    # covering SimGlucose’s 30 patients with multiple 10-day runs, balancing compute and statistical power.

    # Reasoning for '--max_iters': 10 iterations suffice for SimGlucose’s pretraining convergence,
    # reduced from longer runs in the DT paper to save compute for a focused task.

    # Reasoning for '--num_steps_per_iter': 10,000 steps per iteration match the DT paper’s training intensity,
    # ensuring sufficient gradient updates for SimGlucose’s offline phase within fewer iterations.

    # Reasoning for '--device': 'cuda' leverages GPU acceleration for transformer training,
    # critical for SimGlucose’s real-time online tuning efficiency.

    # Reasoning for '--log_to_wandb' ('-w'): False avoids external logging overhead,
    # keeping SimGlucose experiments local and lightweight unless debugging is needed.

    # Reasoning for '--online_training': True enables fine-tuning with new SimGlucose data,
    # critical for adapting pretrained models to real-time patient-specific BG control.

    # Reasoning for '--online_buffer_size': 1000 trajectories balance memory and diversity,
    # keeping top SimGlucose runs for online training, sufficient for patient-specific adaptation.

    # Reasoning for '--save_model': False avoids disk overhead unless explicitly needed,
    # practical for SimGlucose experiments where models are often re-run.

    # Reasoning for '--pretrained_model': None starts training from scratch for SimGlucose,
    # assuming no prior model unless specified, simplifying initial setup.

    # Reasoning for '--stochastic': False ensures deterministic insulin dosing in SimGlucose,
    # prioritizing safety and predictability over exploration in a medical context.

    # Reasoning for '--use_entropy': False skips entropy regularization in SimGlucose,
    # avoiding unnecessary exploration where precise BG control is paramount.

    # Reasoning for '--use_action_means': False samples actions during evaluation,
    # reflecting SimGlucose’s need for realistic variability rather than mean predictions.

    # Reasoning for '--eval_only': False ensures training occurs for SimGlucose,
    # with evaluation as part of the process, not standalone.

    # Reasoning for '--remove_pos_embs': False retains positional embeddings per DT paper,
    # preserving temporal order critical for SimGlucose’s time-series BG data.

    # Reasoning for '--eval_context': None uses training context (K=20) for evaluation,
    # consistent with SimGlucose’s short-term BG prediction needs unless overridden.

    # Reasoning for '--target_entropy': False avoids setting a specific entropy target,
    # simplifying SimGlucose training where stochasticity is deprioritized.

    # Reasoning for '--stochastic_tanh': False skips tanh squashing for actions,
    # allowing SimGlucose’s insulin doses to reflect raw model outputs for flexibility.

    # Reasoning for '--approximate_entropy_samples': 1000 samples provide a reasonable entropy estimate if used,
    # sufficient for SimGlucose’s optional stochastic tuning without excessive compute.
    parser = argparse.ArgumentParser()
    
    #OfflineParams
    parser.add_argument('--env', type=str, default='simglucose')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=int(1e3))
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=int(1e4))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    
    #OnlineParams
    parser.add_argument('--online_training', default=False, action='store_true')
    parser.add_argument('--online_buffer_size', default=1000, type=int) # keep top N trajectories for online training in replay buffer to start
    parser.add_argument('--pretrained_model', default=None, type=str) # './models/2025-02-27_14-07-22/DT_PPO_adolescent#001.pth'
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--stochastic', default=False, action='store_true')
    parser.add_argument('--use_entropy', default=False, action='store_true')
    parser.add_argument('--use_action_means', default=False, action='store_true')
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--remove_pos_embs', default=False, action='store_true')
    parser.add_argument('--eval_context', default=None, type=int)
    parser.add_argument('--target_entropy', default=False, action='store_true')
    parser.add_argument('--stochastic_tanh', default=False, action='store_true')
    parser.add_argument('--approximate_entropy_samples',default=1000, type=int, 
                        help="if using stochastic network w/ tanh squashing, have to approximate entropy with k samples, as no anlytical solution")
    
    
    default_patient = 'adolescent#001'
    algo = 'PPO'
    default_path = f'/home/guleserhocam/VS_Projects/simglucose/dataset/T1DatasetAnalysis/{algo}/output/{default_patient}/{default_patient}_combined_seed.pkl'

    parser.add_argument('--datapath', type=str, default=default_path)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--patient_name', type=str, default=f'{default_patient}')
    parser.add_argument('--algo', type=str, default=f'{algo}')
    parser.add_argument('--savename', type=str, default=f'DT_{algo}_{default_patient}')
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))