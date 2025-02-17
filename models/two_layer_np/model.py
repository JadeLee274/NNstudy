import numpy as np
import nn

class MLP(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int) -> None:
        super(MLP, self).__init__()
        self.in_featrues = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features = self.in_featrues, 
                             out_features = 8, 
                             bias = False)
        self.fc2 = nn.Linear(in_features = self.fc1.out_fearues, 
                             out_features = self.out_features, 
                             bias = False)
        self.activation = nn.relu()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = self.fc1(x)
        self.a1 = self.activation(self.z1)
        self.z2 = self.fc2(self.a1)
        return self.z2
    
    def backward(self, grad_output: np.ndarray) -> None:
        grad_a1 = self.fc2.backward(grad_output = grad_output)
        grad_z1 = grad_a1 * self.activation.grad()
        self.fc1.backward(grad_output = grad_z1)

    def step(self, lr: float) -> None:
        self.fc2.step(lr = lr)
        self.fc1.step(lr = lr)