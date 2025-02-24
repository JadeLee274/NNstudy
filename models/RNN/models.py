import numpy as np
from typing import *


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) / sum(np.exp(x)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


class RNN:
    """ Single RNN layer class \n

    * Args: \n
        Wx is the weight multiplied to the input \n
        Wh is the weight multiplied to the hidden state \n
        b is the bias
    """
    def __init__(self, 
                 Wx: np.ndarray, 
                 Wh: np.ndarray, 
                 b: np.ndarray) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

    def __call__(self, 
                 x: np.ndarray, 
                 h_prev: np.ndarray) -> np.ndarray:
        return self.forward(x, h_prev)

    def forward(self, 
                x: np.ndarray, 
                h_prev: np.ndarray) -> np.ndarray:
        """ Forward pass of the single RNN layer \n

        * Args: \n
            x is the input of the RNN layer \n
            h_prev is the previous state h_{t-1} of the hidden state
        """
        Wx, Wh, b = self.params
        tanh = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(tanh)

        self.cache = x, h_prev, h_next
        return h_next
    
    def backward(self, 
                 dh_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Backward pass of a single RNN layer \n

        * Args: \n
            dh_next is the local gradient of the hidden state
        """
        Wx, Wh, _ = self.params
        x, h_prev, h_next = self.cache

        dtanh = dh_next * (1 - h_next ** 2)

        dWh = np.matmul(h_prev.T, dtanh)
        dWx = np.matmul(x.T, dtanh)
        db = np.sum(dtanh, axis = 0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.matmul(dtanh, Wx.T)
        dh_prev = np.matmul(dtanh, Wh.T)

        return dx, dh_prev


class LSTM:
    """ Single LSTM layer class \n

    * Args: \n
        Wx is the weights of gates that are multiplied to the input vector X_{t} \n
        Wh is the weights of gates that are multiplied to the hidden state vector h_{t-1} \n
        b is the biases of gates    
    """
    def __init__(self, 
                 Wx: np.ndarray, 
                 Wh: np.ndarray, 
                 b: np.ndarray) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def __call__(self, 
                 x: np.ndarray, 
                 h_prev: np.ndarray, 
                 c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.forward(x, h_prev, c_prev)

    def forward(self, 
                x: np.ndarray, 
                h_prev: np.ndarray, 
                c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Forward pass of a single LSTM layer \n
        
        * Args: \n
            x is the input vector \n
            h_prev is the previous hidden state h_{t-1} \n
            c_prev is the previout cell state c_{t-1}
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape # N is the batch size, and H is the dimension of the hidden layer

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H : 2*H]
        i = A[:, 2*H : 3*H]
        o = A[:, 3:H : 4*H]

        f = sigmoid(f)
        g = tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = (f * c_prev) + (g * i)
        h_next = tanh(c_next) * o

        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next)
        return h_next, c_next
    
    def backward(self, 
                 dh_next: np.ndarray, 
                 dc_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Backward pass of a single LSTM layer \n

        * Args: \n
            dh_next is the local gradient of hidden state \n
            dc_next is the local gradient of cell state
        """
        Wx, Wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next = self.cache

        # ====================
        # Gate Backpropagation
        # ====================

        tanh_c_next = np.tanh(c_next)
        ds = dh_next * o * (1 - (tanh_c_next ** 2)) + dc_next
        dc_prev = ds * f # the gradient of the previous memory cell

        # output gate backprop
        do = dh_next * tanh_c_next # local grad of h multiplied by tanh(c_{t})
        do *= o * (1 - o) # sigmoid backprop

        # input gate backprop
        di = ds * g
        di *= i * (1 - i) # sigmoid backprop

        # main gate backprop
        dg = ds * i
        dg *= g * (1 - g)

        # forget gate backprop
        df = ds * c_prev
        df *= f * (1 - f)

        # horizontal satck of the weights of forget/main/input/output gates
        dA = np.hstack((df, dg, di, do))

        # =====================
        # Affine Transformation
        # =====================

        # parameter gradient calculation
        dWx = np.matmul(x.T, dA)
        dWh = np.matmul(h_prev.T, dA)
        db = dA.sum(axis = 0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        # input gradient and hidden state gradient calculation
        dx = np.matmul(dA, Wx.T)
        dh_prev = np.matmul(dA, Wh.T)

        return dx, dh_prev, dc_prev
