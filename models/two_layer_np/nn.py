import numpy as np
from abc import abstractmethod


__all__ = {'relu', 
           'leakyrelu', 
           'tanh', 
           'sigmoid',
           'softmax',
           'MSELoss',
           'CrossEntropyLoss', 
           'Linear', 
           'Module'}


class relu:
    def __init__(self) -> None:
        self.vector = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.vector = np.where(x > 0, x, 0)
        return self.vector
    
    def grad(self) -> np.ndarray:
        return np.where(self.vector > 0, 1, 0)


class leakyrelu:
    def __init__(self, alpha: float) -> None:
        self.vector = None
        self.alpha = alpha
    
    def __call__(self, x: np.ndarray) -> None:
        self.vector = np.where(x > 0, x, self.alpha * x)
        return self.vector
    
    def grad(self) -> np.ndarray:
        return np.where(self.vector > 0, 1, self.alpha)


class tanh:
    def __init__(self) -> None:
        self.vector = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.vector = np.tanh(x)
        return self.vector

    def grad(self) -> np.ndarray:
        return (2 / (1 + np.exp(-2 * self.input))) - 1


class sigmoid:
    def __init__(self) -> None:
        self.vector = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.vector = 1 / (1 + np.exp(-x))
        return self.vector
    
    def grad(self) -> None:
        return self.vector * (1 - self.vector)
    

class softmax:
    def __init__(self) -> None:
        self.vector = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.vector = np.exp(x) / np.sum(np.exp(x))
        return self.vector
    
    def grad(self) -> np.ndarray:
        return self.vector * (1 - self.vector)


class MSELoss:
    def __init__(self) -> None:
        self.value = None
    
    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)
    
    def grad(self) -> np.ndarray:
        return 2 * (self.pred - self.target)
    

class CrossEntropyLoss:
    def __init__(self, eps: float = 1e-9) -> None:
        self.value = None
        self.eps = eps

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred + self.eps
        self.target = target
        self.value = -np.mean(np.sum(self.target * np.log(self.pred + self.eps)))
        return self.value

    def grad(self) -> float:
        return -np.mean(self.target / (self.pred + self.eps))


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        self.in_features = in_features
        self.out_fearues = out_features
        self.weight = np.random.randn(in_features, out_features)
        self.bias = np.random.randn(out_features) if bias else None
        self.use_bias = bias

        self.grad_weight = None
        self.bias_weight = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.dot(x, self.weight)
        if self.use_bias:
            self.output = self.output + self.bias
        return self.output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weight = np.dot(self.input.T, grad_output)
        
        if self.bias:
            self.grad_bias = np.sum(grad_output, axis = 0)

        return np.dot(grad_output, self.weight.T)
    
    def step(self, lr: float) -> None:
        self.weight = self.weight - lr * self.grad_weight
        
        if self.use_bias:
            self.bias = self.bias - lr * self.grad_bias
    
    def zero_grad(self) -> None:
        self.grad_weight = None
        self.bias_weight = None


class Module:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, x: np.ndarray):
        pass

    @abstractmethod
    def step(self, lr: float):
        pass

    @abstractmethod
    def zero_grad(self):
        pass