import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy

#FIXME: Environment sorunlarını hallet!

# Neural Network for Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=-1)  # Softmax for discrete actions
        return x

# Neural Network for Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# DDPG Agent (adapted for discrete actions)
class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def act(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state).detach().numpy()[0]
        probs = probs + np.random.normal(0, noise, size=probs.shape)
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()  # Renormalize
        action = np.argmax(probs)  # Choose action with highest probability
        return action

    def store(self, state, action, reward, next_state, done):
        action_one_hot = np.zeros(self.action_dim)
        action_one_hot[action] = 1
        self.memory.append((state, action_one_hot, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # Critic update
        next_action_probs = self.actor_target(next_state)
        next_action = torch.softmax(next_action_probs, dim=-1)  # Softmax for discrete
        target_q = self.critic_target(next_state, next_action)
        target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_probs = self.actor(state)
        actor_loss = -self.critic(state, actor_probs).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Main training loop
env = gym.make('CartPole-v1')  # Fast-loading environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n  # Discrete actions (2 for CartPole)

agent = DDPG(state_dim, action_dim)
episodes = 200  # Fewer episodes since CartPole is simpler
max_steps = 500

for ep in range(episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        episode_reward += reward

        if done or t == max_steps - 1:
            print(f"Episode {ep + 1}, Reward: {episode_reward:.2f}")
            break

    env.render()  # Optional: visualize progress

env.close()