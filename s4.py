

import jax
import jax.numpy as np
from jax.numpy.linalg import inv, matrix_power
from jax.scipy.signal import convolve


def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def scan_SSM(Ab, Bb, Cb, u, x0):
    def step(x_k_1, u_k):
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k

    return jax.lax.scan(step, x0, u)


def run_SSM(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)

    # Run recurrence
    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]


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

