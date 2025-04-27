import torch
import torchvision
from torch import nn


def get_backbone(model_name: str,
                 device: torch.device,
                 radimgnet_classes: int = 116,
                 pretrained_on: str = "ImageNet"):
    """
    Load a pretrained backbone on RadImageNet from path.
    The .pth file must be in the models/backbone/ directory and the name must be model_name.pth.
    """
    # Initialize variables
    backbone = None
    embedding_dim = None
    # Check if a valid model name was passed
    if model_name == "VGG16":
        # Set the embedding dimension according to the model
        embedding_dim = 512
        # Load model structure from torchvision
        model = torchvision.models.vgg16_bn()
        # Load the pretrained weights from torchvision
        weights = torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1
        # Pass the weights to the model
        model = torchvision.models.vgg16_bn(weights=weights)
        backbone = model.features

    return backbone, embedding_dim
