from typing import *
import numpy as np
Vector = np.ndarray
Matrix = np.ndarray


class MatMul:
    def __init__(
            self,
            W: Matrix,
    ) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(
            self,
            x: Vector,
    ) -> Vector:
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(
            self,
            d_out: Vector
    ) -> Vector:
        W, = self.params
        dx = np.dot(d_out, W.T)
        dW = np.dot(self.x.T, d_out)
        self.grads[0][...] = dW
        return dx


def softmax(
        x: Union[Vector, Matrix],
) -> Union[Vector, Matrix]:
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(
        y: Union[Vector, Matrix],
        t: Union[Vector, Matrix],
) -> Union[Vector, Matrix]:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

 
class SoftmaxWithLoss:
    def __init__(
            self,
    ) -> None:
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(
            self,
            x: Vector,
            t: Vector,
    ) -> float:
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(
            self,
            d_out = 1,
    ) -> Vector:
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= d_out
        dx = dx / batch_size

        return dx


class SimpleCBOW:
    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
    ) -> None:
        V, H = vocab_size, hidden_size

        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)

        self.loss_layer = SoftmaxWithLoss()

        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in
        
    def forward(
            self,
            contexts: Matrix,
            target: Matrix,
    ) -> float:
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(
            self,
            d_out: int = 1,
    ) -> None:

        ds = self.loss_layer.backward(d_out)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
