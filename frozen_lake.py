import gymnasium as gym
import numpy as np
import time

is_slippery = True
env = gym.make('FrozenLake-v1', is_slippery=is_slippery)
env.reset()

num_actions = env.action_space.n
num_states = env.observation_space.n

Q = np.zeros((num_states, num_actions))

num_episodes = 100000
max_steps = 80
alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay_rate = 0.999
epsilon_min = 0.01

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

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay_rate

print(Q)

# env_demo = gym.make('FrozenLake-v1', render_mode='human', is_slippery=is_slippery)
env_demo = gym.make('FrozenLake-v1', is_slippery=is_slippery)

win_count = 0

for e in range(100):
    state, _ = env_demo.reset()
    done = False
    for step in range(max_steps):
        # env_demo.render()
        action = np.argmax(Q[state,:])
        new_state, reward, terminated, truncated, info = env_demo.step(action)
        done = terminated or truncated
        if done:
            env_demo.render()
            if reward == 1:
                win_count += 1
                # print('Win!')
                # time.sleep(1)
            else:
                pass
                # print('Lost!')
                # time.sleep(1)
            break
        state = new_state

print(win_count/100)
