import gym
import numpy as np
import pickle
import os
from gym.envs.registration import register
import simglucose

#FIXME: Code has problem

# Register the simglucose environment
register(
    id='simglucose-adolescent1-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#001'}
)

def basal_bolus_policy(bg, basal_rate=0.02, bolus_threshold=180, bolus_dose=0.1):
    """Simple Basal-Bolus policy: basal insulin with bolus correction."""
    # Ensure bg is a scalar
    bg = np.asarray(bg).item() if np.asarray(bg).size == 1 else bg[0]  # Extract scalar value
    insulin = basal_rate
    if bg > bolus_threshold:
        insulin += bolus_dose
    return np.clip(insulin, 0, 30)

def compute_reward(bg):
    """Reward function based on BG levels."""
    # Ensure bg is a scalar
    bg = np.asarray(bg).item() if np.asarray(bg).size == 1 else bg[0]
    if 70 <= bg <= 180:
        return 1.0
    elif bg < 70:
        return -1.0
    elif bg > 350:
        return -2.0
    else:
        return -1.0

def collect_trajectories(num_episodes=200, max_episode_len=480, seed=42):
    """Collect trajectories from simglucose using a Basal-Bolus policy."""
    env = gym.make('simglucose-adolescent1-v0')
    env.seed(seed)
    np.random.seed(seed)

    trajectories = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        # Debug: Print observation structure
        if ep == 0:
            print(f"Initial observation shape: {np.shape(obs)}, value: {obs}")
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': []
        }
        
        for t in range(max_episode_len):
            bg = obs[0]  # Assuming BG is first element; adjust if needed
            
            # Debug: Print bg type and value
            if ep == 0 and t == 0:
                print(f"BG value: {bg}, type: {type(bg)}")
            
            action = basal_bolus_policy(bg)
            next_obs, reward, done, info = env.step([action])  # Action as list/array
            custom_reward = compute_reward(bg)
            
            trajectory['observations'].append(obs)
            trajectory['actions'].append([action])
            trajectory['rewards'].append(custom_reward)
            trajectory['dones'].append(done)
            
            obs = next_obs
            
            if done:
                break
        
        # Convert to numpy arrays
        trajectory['observations'] = np.array(trajectory['observations'])
        trajectory['actions'] = np.array(trajectory['actions'])
        trajectory['rewards'] = np.array(trajectory['rewards'])
        trajectory['dones'] = np.array(trajectory['dones'])
        
        trajectories.append(trajectory)
        
        total_reward = np.sum(trajectory['rewards'])
        print(f"Episode {ep + 1}/{num_episodes}, Steps: {len(trajectory['rewards'])}, Total Reward: {total_reward:.2f}")

    env.close()
    return trajectories

def save_trajectories(trajectories, save_path):
    """Save trajectories to a pickle file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Trajectories saved to {save_path}")

if __name__ == '__main__':
    num_episodes = 200
    max_episode_len = 480
    save_path = 'dataset/test.pkl'
    
    trajectories = collect_trajectories(num_episodes=num_episodes, max_episode_len=max_episode_len)
    save_trajectories(trajectories, save_path)