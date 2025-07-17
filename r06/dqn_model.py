import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv_net = nn.Sequential(
            # example input (1, 84, 84)
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # stride oznacza co ile pikseli przesuwa się filtr
            # output (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # output (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # output (64, 7, 7)
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out_params_number(input_shape)
        self.dense_net = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, net_input: Tensor) -> Tensor:  # net_input: (num_of_batches, color, width, height)
        conv_out = self.conv_net(net_input)
        conv_out_flat = conv_out.view(net_input.size()[0], -1)  # view changes dimensions of a tensor to (num_of_batches, 7*7)
        return self.dense_net(conv_out_flat)

    def _get_conv_out_params_number(self, shape: np.ndarray[None, int]) -> int:  # shape to kształt obrazu wejściowego do sieci
        dummy_input = torch.zeros(1, *shape)  # * to to samo co ... w js. Czyli np. torch.zeros(1, 64, 7, 7)
        dummy_output = self.conv_net(dummy_input)
        network_output_dimensions = dummy_output.size()
        return int(np.prod(network_output_dimensions))  # np.prod multiplies all items in array 1 * 64 * 7 * 7
