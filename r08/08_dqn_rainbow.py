from typing import Tuple, List

import gymnasium as gym
import numpy as np
from ale_py import ALEInterface, roms

import random

import torch
import torch.optim as optim
from gymnasium import Env

from ignite.engine import Engine
from torch import device

from r08 import common
from r08.agent.DQNAgent import DQNAgent
from r08.network.RainbowDQN import RainbowDQN
from r08.network.TargetNet import TargetNet
from r08.actions.ArgmaxActionSelector import ArgmaxActionSelector
from r08.calc_loss_prio import calc_loss_prio
from r08.experience.ExperienceSourceFirstLast import ExperienceSourceFirstLast, ExperienceFirstLast
from r08.experience.PrioReplayBuffer import PrioReplayBuffer
from r08.wrappers import wrap_dqn

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("pong"))
ale.reset_game()


N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


def process_batch(
        engine: Engine,
        batch_data: Tuple[List[ExperienceFirstLast], np.ndarray, np.ndarray]):
    batch, batch_indices, batch_weights = batch_data
    optimizer.zero_grad()
    loss_v, sample_priorities = calc_loss_prio(
        batch,
        batch_weights,
        net,
        tgt_net.target_model,
        gamma=params.gamma ** N_STEPS,
        device=device
    )
    loss_v.backward()
    optimizer.step()
    buffer.update_priorities(batch_indices, sample_priorities)
    if engine.state.iteration % params.target_net_sync == 0:
        tgt_net.sync()
    return {
        "loss": loss_v.item(),
        "beta": buffer.update_beta(engine.state.iteration),
    }


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env: Env = gym.make("PongNoFrameskip-v4")
    env = wrap_dqn(env)
    env.reset(seed=common.SEED)

    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = TargetNet(net)
    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector, device=device)

    exp_source = ExperienceSourceFirstLast(env, agent, params.gamma, N_STEPS)
    buffer = PrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    trainer_engine = Engine(process_batch)
    common.setup_ignite(trainer_engine, params, exp_source)
    trainer_engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
