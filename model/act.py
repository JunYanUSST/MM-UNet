import torch
import torch.nn as nn


class P_TeLU(nn.Module):
    def __init__(self,
                 alpha=1.0, beta=1.0,
                 learnable_alpha=False, learnable_beta=False):
        super().__init__()
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        else:
            self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))

    def forward(self, x):
        out = self.beta * x * torch.tanh(torch.exp(self.alpha * x))
        return out


class TeLU(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))
