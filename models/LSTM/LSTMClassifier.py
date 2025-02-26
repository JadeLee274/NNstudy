import torch
import torch.nn as nn
Tensor = torch.Tensor

class LSTMClassifier(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            output_dim: int = 1,
    ) -> None:
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]
        out = self.fc(hidden)
        return self.sigmoid(out)
