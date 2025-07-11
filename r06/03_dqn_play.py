#!/usr/bin/env python3
import gymnasium as gym
import time
import numpy as np

import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

from ale_py import ALEInterface, roms

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("pong"))
ale.reset_game()

reward = ale.act(0)  # noop

if __name__ == "__main__":
    env = wrappers.make_env(DEFAULT_ENV_NAME, render_mode='human')
    net = dqn_model.DQN(env.observation_space.shape,
                        env.action_space.n)
    state = torch.load("trained_model_v4_mode_1/PongNoFrameskip-v4-best_-19.dat", map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if done:
            break
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Nagroda sumaryczna: %.2f" % total_reward)
    print("Liczba akcji:", c)
