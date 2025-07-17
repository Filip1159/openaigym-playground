import numpy as np
import torch

from r06.ExperienceBuffer import Experience
from r06.dqn_model import DQN


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset(seed=None)
        self.total_reward = 0.0

    @torch.no_grad()  # nie śledź gradientów, bo tu są niepotrzebne, działa szybciej
    def play_step(self, net: DQN, epsilon: float = 0.0, device: torch.device = "cpu") -> float:
        done_reward = None

        if np.random.random() < epsilon:
            selected_action = self.env.action_space.sample()
        else:
            state = np.array([self.state], copy=False)
            state = torch.tensor(state).to(device)
            net_output = net(state)
            _, best_action = torch.max(net_output, dim=1)
            selected_action = int(best_action.item())

        # wykonaj krok w srodowisku
        new_state, reward, terminated, truncated, _ = self.env.step(selected_action)
        is_done = terminated or truncated
        self.total_reward += reward

        exp = Experience(self.state, selected_action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward
