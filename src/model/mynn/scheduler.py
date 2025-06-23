from torch.optim import lr_scheduler as schedulers
import pytorch_warmup as warmup


def get_schedul(scheduler_name):
    if scheduler_name not in ["ReduceLROnPlateau", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR"]:
        raise NotImplementedError(f"Invalid scheduler name: {scheduler_name}")
    if scheduler_name == "ReduceLROnPlateau":
        return schedulers.ReduceLROnPlateau
    elif scheduler_name == "CosineAnnealingLR":
        return schedulers.CosineAnnealingLR
    elif scheduler_name == "CosineAnnealingWarmRestarts":
        return schedulers.CosineAnnealingWarmRestarts
    elif scheduler_name == "StepLR":
        return schedulers.StepLR


def get_warmup_schedul(scheduler_name, optimizer, kwargs):
    if scheduler_name == 'LinearWarmup':
        print('Using LinearWarmup')
        return warmup.LinearWarmup(optimizer, **kwargs)
    elif scheduler_name == 'ExponentialWarmup':
        return warmup.ExponentialWarmup(optimizer, **kwargs)
    else:
        return None