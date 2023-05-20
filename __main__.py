

from functools import partial

import torch
import jax
import jax.numpy as np


@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, labels):
    one_hot_label = jax.nn.one_hot(labels, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def train_step(state, rng, inputs, labels, model):

    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            inputs,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        loss = np.mean(cross_entropy_loss(logits, labels))
        acc = np.mean(compute_accuracy(logits, labels))
        return loss, (logits, acc)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


def main(seed=0,
         epochs=10,
         batch_size=32
         ):

    torch.random.manual_seed(seed)
    key = jax.random.PRNGKey(seed)

    # Create dataset
    trainloader, testloader = create_dataloader(batch_size=batch_size)

    # Create model
    model = create_model()

    # Create state
    state = create_train_state()

    # Main training loop
    for epoch in epochs:



if __name__ == "__main__":
    main()

