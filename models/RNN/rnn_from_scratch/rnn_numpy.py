from functions import *



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