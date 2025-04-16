from typing import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        eps: float = 1e-12,
    ) -> None:
        batch_size, head, length, tensor_dim = key.size()
        key_t = key.transpose(2, 3)
        attention_score = torch.matmul(query, key_t) / math.sqrt(tensor_dim)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -10000)
        
        attention_score = self.softmax(attention_score)
        value = torch.matmul(attention_score, value)
        return value, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert model_dim % num_heads == 0, f"{model_dim} % {num_heads} != zero"
        
        self.num_heads = num_heads
        self.attention = Attention()
        
        self.w_q = nn.Linear(model_dim, model_dim, bias)
        self.w_k = nn.Linear(model_dim, model_dim, bias)
        self.w_v = nn.Linear(model_dim, model_dim, bias)
        self.w_o = nn.Linear(model_dim, model_dim, bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:        
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = self.split_to_batches(q)
        k = self.split_to_batches(k)
        v = self.split_to_batches(v)

        out, _ = self.attention(q, k, v, mask)

        out = self.concat_from_batches(out)
        out = self.w_o(out)
        return out
    
    def split_to_batches(
        self,
        x: Tensor
    ) -> Tensor:
        batch_size, seq_len, model_dim = x.size()
        tensor_dim = model_dim // self.num_heads
        x = x.view(batch_size, seq_len, self.num_heads, tensor_dim)
        x = x.transpose(1, 2)
        return x
    
    def concat_from_batches(
        self,
        x: Tensor
    ) -> Tensor:
        batch_size, head, length, tensor_dim = x.size()
        model_dim = head * tensor_dim
        x = x.transpose(1, 2).contiguous().view(batch_size, length, model_dim)
        return x
