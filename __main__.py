

import torch
import torch.nn.functional as F

import datetime
import matplotlib.pyplot as plt

from dataloader import create_dataloader
from model import Model


def main(epochs=100,
         batch_size=256
         ):

    dt = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    file = open(f"logs/{dt}", "w")

    # Create dataset
    trainloader, testloader = create_dataloader(batch_size=batch_size)

    # Create model
    model = Model(d_output=1,
                  d_model=256,
                  n_layers=4,
                  dropout=0.1,
                  transposed=False).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Main training loop
    for epoch in range(epochs):
        for idx, samples in enumerate(trainloader):
            data, label = samples

            data = data.reshape(batch_size, -1)
            inputs, targets = data[:, :-1].view(batch_size, -1, 1).cuda(), data[:, 1:].view(batch_size, -1, 1).cuda()

            # (batch_size, d_model, length)
            pred = model(inputs)
            loss = F.mse_loss(targets, pred)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            print("loss ", loss.item())

            file.write("{}\n".format(loss))
            file.flush()

            # validation
            # save generated image to images folder
            # task is to finish mnist sequence after looking at 300 pixels
            if idx == 0:
                x = data[0, :300].cuda().view(300)

                ans = []
                states = model.get_state()

                for i in range(300):
                    ans.append(x[i].view(1, 1))
                    y, states = model.step(x[i].view(1, 1), states)

                ans.append(y)

                for i in range(300, 28 * 28 - 1):
                    y, states = model.step(y, states)
                    ans.append(y)

                ans = torch.stack(ans).view(28, 28).detach().cpu().numpy()
                plt.imsave(f"images/{epoch}.png", ans, cmap="gray")


if __name__ == "__main__":
    main()

