# useful functions
from config import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# MSE for Loss
def MSE(y, t):
    return 0.5 * np.mean((y - t) ** 2)
