from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor

from r08.network.NoisyLinear import NoisyLinear


class RainbowDQN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], n_actions: int):
        super(RainbowDQN, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out_params_number(input_shape)
        # calculates quality of states after performing each action
        self.dense_advantage_net = nn.Sequential(
            NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            NoisyLinear(256, n_actions)
        )
        # calculates general quality of the state regardless of next actions
        self.dense_value_net = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, net_input: Tensor) -> Tensor: # net_input: (num_of_batches, color, width, height)
        advantage, value = self.get_advantage_and_value(net_input)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def get_advantage_and_value(self, net_input: Tensor) -> Tuple[Tensor, Tensor]:
        float_net_input = net_input.float() / 256
        conv_out = self.conv_net(float_net_input)
        conv_out_flat = conv_out.view(float_net_input.size()[0], -1)
        return self.dense_advantage_net(conv_out_flat), self.dense_value_net(conv_out_flat)

    def _get_conv_out_params_number(self, shape: np.ndarray[None, int]) -> int:  # shape to kształt obrazu wejściowego do sieci
        dummy_input = torch.zeros(1, *shape)  # * to to samo co ... w js. Czyli np. torch.zeros(1, 64, 7, 7)
        dummy_output = self.conv_net(dummy_input)
        network_output_dimensions = dummy_output.size()
        return int(np.prod(network_output_dimensions))  # np.prod multiplies all items in array 1 * 64 * 7 * 7
