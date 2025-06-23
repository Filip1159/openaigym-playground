import time

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"Using device: {device}")

# Konfiguracja
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000  # liczba kroków

# Prosty model sieci DQN
class DQN(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# Funkcja wybierająca akcję zgodnie z epsilon-greedy
def select_action(model, state, epsilon, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_v = torch.tensor([state], dtype=torch.float32).to(device)
        q_vals = model(state_v)
        _, act_v = torch.max(q_vals, dim=1)
        return int(act_v.item())

# Główna pętla
env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

replay_buffer = deque(maxlen=BUFFER_SIZE)
model = DQN(obs_size, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

epsilon = EPSILON_START
total_rewards = []
state, _ = env.reset()
episode_reward = 0
frame_idx = 0

while True:
    frame_idx += 1
    epsilon = max(EPSILON_END, EPSILON_START - frame_idx / EPSILON_DECAY)

    action = select_action(model, state, epsilon, env)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    replay_buffer.append((state, action, reward, next_state, done))

    state = next_state
    episode_reward += reward

    if done:
        state, _ = env.reset()
        total_rewards.append(episode_reward)
        print(f"Episode {len(total_rewards)}: reward={episode_reward}")
        episode_reward = 0

    # Czekamy aż bufor się wypełni
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        continue

    # Pobieramy mini-batch
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones_t = torch.tensor(dones, dtype=torch.bool).to(device)

    # Obliczanie Q(s,a)
    state_action_values = model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Obliczanie max Q(s', a') dla stanu następnego
    with torch.no_grad():
        next_state_values = model(next_states_v).max(1)[0]
        next_state_values[dones_t] = 0.0
        expected_values = rewards_v + GAMMA * next_state_values

    # Funkcja straty i optymalizacja
    loss = nn.MSELoss()(state_action_values, expected_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if np.mean(total_rewards[-100:]) > 195:
        print(f"Solved in {len(total_rewards)} episodes!")
        break

env_demo = gym.make("CartPole-v1", render_mode='human')
state, _ = env_demo.reset()
env_demo.render()

while True:
    state_v = torch.tensor([state], dtype=torch.float32).to(device)
    q_vals = model(state_v)
    _, act_v = torch.max(q_vals, dim=1)
    action = int(act_v.item())

    next_state, reward, terminated, truncated, _ = env_demo.step(action)
    done = terminated or truncated
    env_demo.render()

    state = next_state

    if done:
        break
