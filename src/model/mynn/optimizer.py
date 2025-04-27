import torch.optim as optim
from lion_pytorch import Lion


def get_optim(optimizer_name):
    """
    Get the optimizer class based on the specified name.
    """
    if optimizer_name == "Adam":
        return optim.Adam
    elif optimizer_name == "AdamW":
        return optim.AdamW
    elif optimizer_name == "SGD":
        return optim.SGD
    elif optimizer_name == "Lion":
        return Lion
    else:
        raise NotImplementedError(f"Invalid optimizer name: {optimizer_name}")