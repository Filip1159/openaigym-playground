from typing import Tuple, List, Union

import numpy as np
import torch
from torch import nn, Tensor

GAMMA = 0.99


def calc_loss(
        batch: Tuple[
            List[np.ndarray],  # states
            List[int],  # actions
            List[float],  # rewards
            List[bool],  # dones
            List[np.ndarray]  # next_states
        ],
        net: nn.Module,
        tgt_net: nn.Module,
        device: Union[str, torch.device] = "cpu"
) -> Tensor:
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).type('torch.LongTensor').to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # 1. mamy listę stanów środowiska (jako tensor)
    # 2. wrzucamy tensor do sieci, a ona zwraca Q dla każdej z możliwych akcji
    # 3. wybieramy tę akcję, którą została zapisana w batchu Experience

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        # weż najlepszą akcję dla next_states_v .........
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0  # reward = 0 where episode has ended
        next_state_values = next_state_values.detach()  # detach() is probably not necessary, thanks to no_grad()

    expected_state_action_values = next_state_values * GAMMA + rewards_v

    # state_action_values is from trained net (state -> value)
    # expected_state_action_values is from target net (reward + next_state -> value)
    return nn.MSELoss()(state_action_values, expected_state_action_values)
