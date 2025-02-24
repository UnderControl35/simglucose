import gym
import numpy as np
import torch
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
import sys, os

import simglucose
from gym.envs.registration import register

register(
    id='simglucose-adolescent1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#001'}
)

def test_decision_transformer(model_path="dt_test.pth", env_name="hopper", num_episodes=10):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if env_name == 'simglucose':
        env = gym.make('simglucose-adolescent1-v0')
        max_ep_len = 480
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        env_targets = [180, 70]
        scale = 1000.
    # Environment setup
    elif env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]  # Target return for evaluation
        scale = 1000.
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment {env_name} not supported")

    K = 20

    # Define model architecture (must match your trained model)
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,  # Context length K - adjust if different
        max_ep_len=max_ep_len,
        hidden_size=128,  # Adjust to match your training
        n_layer=3,        # Adjust to match your training
        n_head=4,         # Adjust to match your training
        n_inner=4*128,    # Typically 4*hidden_size
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,  # Adjust to match your training
        attn_pdrop=0.1    # Adjust to match your training
    )

    model_path_file = os.path.join(model_path, "dt_simglucose.pth")
    # Load the saved state dictionary
    state_dict = torch.load(model_path_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # These should match your training setup - adjust if needed
    #state_mean = 0.0  # You might need to load actual values used in training
    #state_std = 1.0   # You might need to load actual values used in training
    
    #FIXME: Move to the file into arguments
    load_dir = model_path
    # If you saved these during training, load them instead
    state_mean = np.load(os.path.join(load_dir, "state_mean.npy"))
    state_std = np.load(os.path.join(load_dir, "state_std.npy"))

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
                    mode='normal',  # Adjust if you used 'delayed' mode
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                )
            returns.append(ret)
            lengths.append(length)
            print(f"Episode {episode + 1}: Return = {ret:.2f}, Length = {length}")
        
        return returns, lengths

    # Run evaluation
    print(f"\nTesting model on {env_name} environment")
    print("=" * 50)
    
    for target in env_targets:
        print(f"\nEvaluating with target return: {target}")
        returns, lengths = eval_episode(target)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(lengths)
        
        print(f"\nResults for target {target}:")
        print(f"Average Return: {mean_return:.2f} ± {std_return:.2f}")
        print(f"Average Episode Length: {mean_length:.2f}")
        print(f"Success Rate (>= Target): {np.mean(np.array(returns) >= target)*100:.1f}%")

if __name__ == "__main__":
    # You can modify these parameters
    test_decision_transformer(
        model_path="models/2025-02-24_17-58-50",
        env_name="simglucose",  # Change to your environment
        num_episodes=10     # Number of test episodes
    )