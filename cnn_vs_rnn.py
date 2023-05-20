

import jax
import jax.numpy as np
from jax.numpy.linalg import matrix_power
from jax.scipy.signal import convolve

from s4 import discretize, run_SSM


rng = jax.random.PRNGKey(1)


def K_conv(Ab, Bb, Cb, L):
    """
    Creates a giant filter for convolution mode of s4

    Equation:
        K = (CB, CAB, CAAB, ..., CA^(L-1)B
    """
    return np.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).reshape() for l in range(L)]
    )


def causal_convolution(u, K, nofft=False):
    if nofft:
        return convolve(u, K, model="full")[:, u.shape[0]]
    else:
        assert K.shape[0] == u.shape[0]

        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd

        return np.fft.irfft(out)[: u.shape[0]]


def random_SSM(rng, N):
    a_r, b_r, c_r = jax.random.split(rng, 3)

    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))

    return A, B, C


def test_cnn_is_rnn(N=4, L=16, step=1.0 / 16):
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)

    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    print("u ", u.shape)
    conv = causal_convolution(u, K_conv(*ssmb, L))
    print("conv ", conv.shape)

    # Check
    assert np.allclose(rec.ravel(), conv.ravel())


if __name__ == "__main__":
    test_cnn_is_rnn()
