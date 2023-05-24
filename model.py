

import torch
import torch.nn as nn

from s4 import S4Block
from s4d import S4D


class Model(nn.Module):

    def __init__(self, d_output, d_model, n_layers, dropout, transposed):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(d_output, d_model),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_output),
            nn.Sigmoid()
        )
        self.layers = [
            S4Block(
                d_model=d_model,
                dropout=dropout,
                transposed=transposed,
                final_act="glu"
            )
            for _ in range(n_layers)
        ]
        # self.layers = [
        #     S4D(
        #         d_model=d_model,
        #         dropout=dropout,
        #         transposed=transposed
        #     )
        #     for _ in range(n_layers)
        # ]

    def setup(self):
        for layer in self.layers:
            layer.cuda()
            layer.setup_step()

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x, state = layer(x)
        x = self.decoder(x)
        return x

    @torch.no_grad()
    def step(self, x, states):
        x = self.encoder(x)

        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer.step(x, state)
            new_states.append(state)
        x = self.decoder(x)
        return x, new_states

    def get_state(self):
        new_states = []
        for layer in self.layers:
            state = layer.default_state()
            new_states.append(state)

        return new_states

