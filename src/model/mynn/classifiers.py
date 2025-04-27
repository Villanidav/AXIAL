import torch
from torch import nn


def get_classifier(embedding_dim, num_classes, dropout, hidden_dim=0, num_hidden_layers=0):
    """
    Get a classifier based on the specified parameters.

    """
    if num_hidden_layers == 0:
        # Return a fully connected layer only using dropout
        return nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, num_classes)
        )
    else:
        # Return a fully connected layer with hidden layers ( Linear -> ReLU -> Dropout )
        return nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ModuleList([nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout, inplace=True)
            ) for _ in range(num_hidden_layers)]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
