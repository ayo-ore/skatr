import torch.nn as nn
import torch.nn.functional as F
from itertools import pairwise
from typing import List, Optional
from omegaconf import DictConfig


class MLP(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        act: str = "relu",
        out_act: Optional[str] = None,
        drop: float = 0.0,
    ):

        # units, act, drop=None

        super(MLP, self).__init__()

        self.out_channels = out_channels
        units = [in_channels, *hidden_channels, out_channels]
        self.linear_layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in pairwise(units)]
        )
        self.act = getattr(F, act)
        self.out_act = getattr(F, out_act) if out_act else None
        self.drop = nn.Dropout(drop) if drop else None

    def forward(self, x):

        for linear in self.linear_layers[:-1]:

            x = linear(x)
            x = self.act(x)
            if self.drop is not None:
                x = self.drop(x)

        x = self.linear_layers[-1](x)
        if self.out_act is not None:
            x = self.out_act(x)

        return x
