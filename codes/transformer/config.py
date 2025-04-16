import torch


# GPU device setting
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


# Model parameter setting
batch_size = 512
max_len = 256
model_dim = 512
num_layers = 6
num_heads = 8
feedforward_dim = 2048
dropout = 0.1


# Optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')