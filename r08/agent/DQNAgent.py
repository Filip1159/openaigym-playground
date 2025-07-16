import numpy as np
import torch
from gymnasium.core import ObsType, ActType

from r08.actions.ArgmaxActionSelector import ArgmaxActionSelector


class DQNAgent:
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector: ArgmaxActionSelector, device="cpu"):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.device = device

    def __call__(self, state: ObsType) -> ActType:
        state = torch.tensor(np.expand_dims(state, 0)).to(self.device)
        q_v = self.dqn_model(state)
        q = q_v.data.cpu().numpy()
        return self.action_selector(q)[0]
