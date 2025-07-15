import gymnasium as gym
from ale_py import ALEInterface, roms

import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from r08 import wrappers as ptan_wrappers, common, dqn_extra
from r08.agent.DQNAgent import DQNAgent
from r08.agent.TargetNet import TargetNet
from r08.actions.ArgmaxActionSelector import ArgmaxActionSelector
from r08.calc_loss_prio import calc_loss_prio
from r08.experience.ExperienceSourceFirstLast import ExperienceSourceFirstLast
from r08.experience.PrioReplayBuffer import PrioReplayBuffer

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("pong"))
ale.reset_game()


N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("PongNoFrameskip-v4")
    env = ptan_wrappers.wrap_dqn(env)
    env.reset(seed=common.SEED)

    net = dqn_extra.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = TargetNet(net)
    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector, device=device)

    exp_source = ExperienceSourceFirstLast(env, agent, params.gamma, N_STEPS)
    buffer = PrioReplayBuffer(exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss_prio(
            batch, batch_weights, net, tgt_net.target_model,
            gamma=params.gamma**N_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "beta": buffer.update_beta(engine.state.iteration),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source)
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
