import math

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, sigma_init: int = 0.017, bias: bool = True):

        # sigma - trainable parameter representing the amount of noise
        #     (more significant at the beginning, at the end -> 0)
        # epsilon - non-trainable random noise source

        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_weight_init: Tensor = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(sigma_weight_init)
        epsilon_weight_buffer = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight_buffer)
        if bias:
            sigma_bias_init = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(sigma_bias_init)
            epsilon_bias_buffer = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias_buffer)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, net_input: Tensor) -> Tensor:
        self.epsilon_weight.normal_()  # override tensor with random std values (inplace)
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()  # override tensor with random std values (inplace)
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        weight_with_noise = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(net_input, weight_with_noise, bias)
