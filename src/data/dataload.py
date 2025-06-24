from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from src.data.dataset import ADNIDataset
from torchvision import transforms
from typing import Tuple, Any, Optional
import pandas as pd
from transform import RandomTransformations


def get_dataloaders(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    batch_size: int,
                    num_workers: int,
                    num_slices: int,
                    train_transform: transforms.Compose,
                    test_transform: transforms.Compose,
                    data_augmentation: Optional[RandomTransformations] = None,
                    data_augmentation_slice: Optional[transforms.Compose] = None,
                    revert_slices_order: bool = False,
                    slicing_plane: str = "axial",
                    rank: int = 0,
                    world_size: int = 1,
                    seed: int = 42) -> \
        Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], int]:
    """
    Create dataloaders for train, validation and test sets (modified for DDP).
    The dataloaders iterate over the dataset and return a batch of images and labels.
    """
    is_distributed = world_size > 1
    is_main_process = rank == 0

    # Create classes and class_to_idx attributes
    classes = sorted(train_df.diagnosis.unique())
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    # Show the index of each class (only on main process)
    if is_main_process:
        print(f"Class to index: {class_to_idx}", flush=True)

    # Create the datasets
    train_dataset = None
    val_dataset = None
    train_dataset = ADNIDataset(dataframe=train_df,
                                transform=train_transform,
                                data_augmentation=data_augmentation,
                                data_augmentation_slice=data_augmentation_slice,
                                num_slices=num_slices,
                                classes=classes,
                                class_to_idx=class_to_idx,
                                revert_slices_order=revert_slices_order,
                                slicing_plane=slicing_plane)
                                
    val_dataset = ADNIDataset(dataframe=val_df,
                                transform=test_transform,
                                num_slices=num_slices,
                                classes=classes,
                                class_to_idx=class_to_idx,
                                revert_slices_order=revert_slices_order,
                                slicing_plane=slicing_plane)

    test_dataset = ADNIDataset(dataframe=test_df,
                               transform=test_transform,
                               num_slices=num_slices,
                               classes=classes,
                               class_to_idx=class_to_idx,
                               revert_slices_order=revert_slices_order,
                               slicing_plane=slicing_plane)

    if train_dataset is None or val_dataset is None or test_dataset is None:
        raise ValueError("Datasets must not be None before creating samplers or dataloaders.")

    # --- Create DistributedSamplers ---
    train_sampler = None
    val_sampler = None
    test_sampler = None
    train_shuffle = not is_distributed # Shuffle with DataLoader only if not distributed

    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True, # Shuffle training data across ranks
            seed=seed     # Use provided seed for consistent shuffling
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False # No shuffling for validation
        )
        # Use sampler for test set as well to ensure non-overlapping evaluation
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False # No shuffling for test
        )

    # --- Create DataLoaders ---
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, # Per-GPU batch size
        shuffle=train_shuffle, # Must be False if sampler is used
        num_workers=num_workers,
        sampler=train_sampler, # Pass the sampler
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=is_distributed # Drop last non-full batch in distributed mode
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size, # Per-GPU batch size (can be same or different)
        shuffle=False,
        num_workers=num_workers,
        sampler=val_sampler, # Pass the sampler
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False # Usually keep all validation samples
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size, # Use adjusted or original batch size
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler, # Pass the sampler
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False # Keep all test samples
    )

    # Print information (only on main process)
    if is_main_process:
        print(f"\nDistributed Training Enabled: {is_distributed} (World Size: {world_size})")
        print(f"Per-GPU Batch Size: {batch_size}")
        print(f"Effective Total Batch Size: {batch_size * world_size}")
        print(f"\nClasses: {classes}\n")
        print(f"\nTraining classes distribution:\n{train_df['diagnosis'].value_counts()}")
        print(f"\nValidation samples distribution:\n{val_df['diagnosis'].value_counts()}")
        print(f"\nTest samples distribution:\n{test_df['diagnosis'].value_counts()}")
        print(f"\nTrain dataset samples: {len(train_dataset)}")
        print(f"Validation dataset samples: {len(val_dataset)}")
        print(f"Test dataset samples: {len(test_dataset)}")
        # Note: len(dataloader) shows batches per process
        print(f"\nTrain batches per process: {len(train_dataloader)}")
        print(f"Validation batches per process: {len(val_dataloader)}")
        print(f"Test batches per process: {len(test_dataloader)}\n")
        # Print the shape of the images from the first batch of this process
        try:
            first_batch_data, _ = next(iter(train_dataloader))
            print(f"\nTrain images shape (this process): {first_batch_data.shape}", flush=True)
        except StopIteration:
            print("\nCould not retrieve shape from train_dataloader (possibly empty for this rank).", flush=True)

    return train_dataloader, val_dataloader, test_dataloader, len(classes)
