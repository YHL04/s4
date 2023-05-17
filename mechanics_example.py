
from functools import partial
import numpy as np


def example_mass(k, b, m):
    A = np.array([[0, 1],
                  [-k/m, -b/m]])

    B = np.array([[0],
                  [1.0 / m]])

    C = np.array([[1.0, 0]])

    return A, B, C


@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)


