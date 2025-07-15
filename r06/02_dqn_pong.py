#!/usr/bin/env python3
from r06 import dqn_model, wrappers

import time
import numpy as np

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from ale_py import ALEInterface, roms

from r06.Agent import Agent
from r06.ExperienceBuffer import ExperienceBuffer
from r06.calc_loss import calc_loss

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("pong"))
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    env = wrappers.make_env(DEFAULT_ENV_NAME)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    episode_start_frame = 0    # last frame of last finished episode
    episode_start_timestamp = time.time()    # timestamp of last finished episode
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device)
        if reward is not None:  # episode has ended
            total_rewards.append(reward)
            speed = (frame_idx - episode_start_frame) / (time.time() - episode_start_timestamp)
            episode_start_frame = frame_idx
            episode_start_timestamp = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: gry - %d, nagroda %.3f, eps %.2f, %.2f fps" % (frame_idx, len(total_rewards), m_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Nagroda ulegla zmianie: %.3f -> %.3f" % (best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Rozwiazano po %d klatkach!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()
    writer.close()
