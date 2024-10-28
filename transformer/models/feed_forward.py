import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
