import torch
import torch.nn as nn
import torch.nn.functional as func


class AttentionLayer(nn.Module):
    """
    Attention module. This module is used to compute attention weights and apply them to the input.
    """

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, 1)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        # Compute attention scores
        scores = self.linear(inputs).view(batch_size, seq_len)
        # Apply softmax to get attention weights
        attention_weights = func.softmax(scores, dim=1)
        # Compute the weighted mean
        weighted_mean = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return weighted_mean, attention_weights
