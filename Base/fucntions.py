# useful functions
from config import *


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# MSE for Loss
def MSE(y, t):
    return 0.5 * np.mean((y - t) ** 2)
