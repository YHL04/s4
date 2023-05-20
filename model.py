

from flax import linen as nn
from layer import S4Layer


class SequenceBlock(nn.Module):
    dropout: float
    d_model: int

    def setup(self):
        self.seq = S4Layer()
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0])

    def __call__(self, x):
        skip = x
        x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        x = self.out(x)
        x = skip + self.drop(x)

        return x


class Model(nn.Module):
    d_output: int
    d_model: int
    n_layers: int
    dropout: float = 0.0

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                d_model=self.d_model,
                dropout=self.dropout
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


if __name__ == "__main__":
    import torch

    model = Model(d_output=1, d_model=64, n_layers=4)

    x = torch.rand(3, 3)
    print(model(x))
