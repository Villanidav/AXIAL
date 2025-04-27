from torch import nn


class BackboneWithClassifier(nn.Module):
    """
    This module is used to attach a classifier to a backbone network.
    """

    def __init__(self, backbone, embedding_dim, num_classes, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
