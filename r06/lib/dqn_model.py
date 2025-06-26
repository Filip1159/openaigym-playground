import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv = nn.Sequential(
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

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):  # shape to kształt obrazu wejściowego do sieci
        o = self.conv(torch.zeros(1, *shape))  # * to to samo co ... w js. Czyli np. torch.zeros(1, 64, 7, 7)
        return int(np.prod(o.size()))  # np.prod multiplies all items in array 1 * 64 * 7 * 7

    def forward(self, x):  # x: (num_of_batches, color, width, height)
        conv_out = self.conv(x).view(x.size()[0], -1)  # view changes dimensions of a tensor to (num_of_batches, 7*7)
        return self.fc(conv_out)
