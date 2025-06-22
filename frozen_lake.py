import gymnasium as gym
import numpy as np
import time
from IPython.display import display

env = gym.make('FrozenLake-v1')
print(env.metadata['render_modes'])
env.reset()
# env.render()

num_actions = env.action_space.n
num_states = env.observation_space.n

Q = np.zeros((num_states, num_actions))
print(Q)

num_episodes = 10000
max_steps = 80
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_decay_rate = 0.0001

for e in range(num_episodes):
    state, _ = env.reset()
    done = False
    for step in range(max_steps):
        if np.random.random() > epsilon:  # exploit
            action = np.argmax(Q[state])
        else:  # explore
            action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        Q[state, action] = Q[state, action] + alpha * (float(reward) + gamma * float(np.max(Q[next_state,:])) - Q[state, action])
        state = next_state
        if done:
            break

    epsilon = epsilon * np.exp(-epsilon_decay_rate * e)

print(Q)

env_demo = gym.make('FrozenLake-v1', render_mode='human')

for e in range(4):
    state, _ = env_demo.reset()
    done = False
    for step in range(max_steps):
        env_demo.render()
        action = np.argmax(Q[state,:])
        new_state, reward, terminated, truncated, info = env_demo.step(action)
        done = terminated or truncated
        if done:
            env_demo.render()
            if reward == 1:
                print('Win!')
                time.sleep(1)
            else:
                print('Lost!')
                time.sleep(1)
            break
        state = new_state
