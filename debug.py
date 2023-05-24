
import torch

x = torch.arange(9).view(3, 3)
print(x)
print(torch.einsum("h n -> n h", x))


