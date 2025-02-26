import numpy as np
from typing import *


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) / sum(np.exp(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))