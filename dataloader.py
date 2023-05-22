

import torch
from torchvision import transforms
from torchvision.datasets import MNIST


def create_dataloader(batch_size):
    data_train = torch.utils.data.DataLoader(
        MNIST(
            '~/mnist_data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=batch_size,
        shuffle=True
    )

    data_test = torch.utils.data.DataLoader(
        MNIST(
            '~/mnist_data', train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=batch_size,
        shuffle=True
    )

    return data_train, data_test


# data_train, data_test = create_dataloader()
#
# for batch_idx, samples in enumerate(data_train):
#     train, test = samples
#
#     print(train.view(64, -1).shape)
#     print(test.shape)
