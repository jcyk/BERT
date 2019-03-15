import torch
from torch import nn
from torch.nn import Parameter

import math

def gelu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class LayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.Tensor(hidden_size))
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
            self.eps = eps
            self.reset_parameters()
        def reset_parameters(self):
            nn.init.constant_(self.weight, 1.)
            nn.init.constant_(self.bias, 0.)

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight * x + self.bias
