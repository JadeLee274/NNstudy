from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor


__all__ = ['Attention', 'MultiHeadAttention']


class Attention(nn.Module):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask = None
    ) -> Tensor:
        d_k = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1))
        scores /= torch.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
        attn = torch.softmax(scores, dim=-1)
        return attn.matmul(value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        bias: bool = True,
        activation = F.relu,
    ) -> None:
        """
        :param in_features: Size of each input sample
        :param num_heads  : Number of heads
        :param bias       : Whether to use bias
        :param activation : Activation function after each transformation
        """
        super().__init__()
        if in_features % num_heads != 0:
            raise ValueError(
                f'in_features {in_features} must be divisible by num_heads {num_heads}'
            )
        self.in_features = in_features
        self.num_heads = num_heads
        self.activation = activation
        self.bias = bias
        self.w_q = nn.Linear(in_features, in_features, bias)
        self.w_k = nn.Linear(in_features, in_features, bias)
        self.w_v = nn.Linear(in_features, in_features, bias)
        self.w_o = nn.Linear(in_features, in_features, bias)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask = None,
    ) -> Tensor:
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        
        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        
        out = Attention()(q, k, v, mask)
        out = self._reshape_from_batches(out)
        out = self.w_o(out)
        
        if self.activation is not None:
            out = self.activation(out)
        
        return out
    
    def _reshape_to_batches(
        self,
        x: Tensor,
    ) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        x = x.reshape(batch_size, seq_len, self.num_heads, sub_dim)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * self.num_heads, seq_len, sub_dim)
        return x
    
    def _reshape_from_batches(
        self,
        x: Tensor
    ) -> Tensor:
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        x = x.reshape(batch_size, self.num_heads, seq_len, in_feature)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, seq_len, out_dim)
        return x
    
    @staticmethod
    def generate_history_mask(x: Tensor) -> Tensor:
        """
        Generate mask
        """
        batch_size, seq_len, _ = x.size()
        mask = torch.ones(seq_len, seq_len)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1)
        return torch.tril(mask)
    
    def extra_repr(self):
        return (
            f'in_features={self.in_features}, num_heads={self.num_heads}, ', 
            f'bias={self.bias}, activation={self.activation}'
        )
    
