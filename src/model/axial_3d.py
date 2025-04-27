import torch.nn as nn
import mynn


class Axial3D(nn.Module):
    """
    Attentional Convolutional 3D Network. This module is used to classify 3D images. 
    It uses a backbone to extract features from the images and then applies an attentional layer to the features. 
    Finally, it applies a classifier to the output of the attentional layer.
    """

    def __init__(self, backbone, embedding_dim, num_slices, num_classes, dropout=0.4, return_attention_weights=False):
        super().__init__()
        self.num_slices = num_slices
        self.feat_map_dim = embedding_dim
        self.backbone = backbone
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.attention = mynn.AttentionLayer(input_size=embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        self.return_attention_weights = return_attention_weights

    def forward(self, x):
        # ---------------------- x.shape = (batch_size, num_slices, 3, 224, 224) ----------------------------------
        # 1. Reshape input to combine batch and images dimensions to create a single batch of images.
        x = x.view(-1, *x.size()[2:])  # e.g. (32, 80, 3, 224, 224) -> (32 * 80, 3, 224, 224)
        # 2. Pass the input through the backbone
        x = self.backbone(x)  # e.g. (32 * 80, 3, 224, 224) -> (32 * 80, 1280, 7, 7)
        # 3. Apply the AdaptiveAvgPool2d
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)  # e.g. (32 * 80, 1280, 1, 1) -> (32 * 80, 1280)
        # 4. Turn back the batch dimension to separate the 3D images
        x = x.view(-1, self.num_slices, *x.size()[1:])  # e.g. (32 * 80, 1280) -> (32, 80, 1280)
        # 5. Compute the attention
        x, attention_weights = self.attention(x)  # e.g. (32, 80, 1280) -> (32, 1280)
        # 6. Apply the classifier
        x = self.classifier(x)  # e.g. (32, 1280) -> (32, num_classes)
        if self.return_attention_weights:
            return x, attention_weights
        return x
