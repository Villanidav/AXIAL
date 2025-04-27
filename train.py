from datetime import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from src.models.aware_net import run_aware_experiment
from src.data import train_val_test_subject_split
from src.models import compute_metrics
from src.logger import create_writer
from src.data import load_dataframe
from src.config import load_config
from src.utils import get_device
from src.utils import set_seeds
from src import experiments
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initializes the distributed process group based on environment variables."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Print environment variables for debugging (can be removed later)
    print(f"PID: {os.getpid()} | RANK env: {os.environ.get('RANK')}", flush=True)
    print(f"PID: {os.getpid()} | LOCAL_RANK env: {os.environ.get('LOCAL_RANK')}", flush=True)
    print(f"PID: {os.getpid()} | WORLD_SIZE env: {os.environ.get('WORLD_SIZE')}", flush=True)
    print(f"PID: {os.getpid()} | MASTER_ADDR env: {os.environ.get('MASTER_ADDR')}", flush=True)
    print(f"PID: {os.getpid()} | MASTER_PORT env: {os.environ.get('MASTER_PORT')}", flush=True)
    print(f"PID: {os.getpid()} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    # We expect this count to be 1 now due to CUDA_VISIBLE_DEVICES set by torchrun
    print(f"PID: {os.getpid()} | CUDA Device Count (visible): {torch.cuda.device_count()}", flush=True)

    # --- MODIFIED CONDITION ---
    # Initialize if world_size > 1 (meaning it's a distributed launch)
    # and CUDA is available. torchrun handles GPU assignment via CUDA_VISIBLE_DEVICES.
    if world_size > 1 and dist.is_available() and torch.cuda.is_available():
        # The backend='nccl' or 'gloo' can often be automatically chosen
        # but explicitly setting 'nccl' is standard for NVIDIA GPUs.
        # torchrun sets MASTER_ADDR/PORT, so init_method='env://' works implicitly.
        dist.init_process_group(backend='nccl')
        # Re-fetch rank/world_size AFTER init_process_group for verification
        # although os.environ should be correct from torchrun
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # local_rank is typically still correctly obtained from os.environ["LOCAL_RANK"]
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Rank {rank}/{world_size} | Local Rank {local_rank} | Successfully initialized process group (backend='nccl').", flush=True)
        return rank, world_size, local_rank
    else:
        print(f"Rank {rank}/{world_size} | Not initializing distributed mode (world_size={world_size}, dist_available={dist.is_available()}, cuda_available={torch.cuda.is_available()}). Running in single-process mode.", flush=True)
        # Return defaults for single-process execution
        return 0, 1, 0


def cleanup_distributed():
    """Cleans up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up process group.", flush=True)


def main():
    # --- Distributed Setup ---
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

    # Get the timestamp for the experiment (only on main process for consistency)
    if is_main_process:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        timestamp = None
    # Broadcast timestamp from rank 0 to all other processes
    timestamp = broadcast_object(timestamp, rank, device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))


    # Load the yaml config file
    config = load_config('config.yaml')

    # --- Device Setup ---
    if torch.cuda.is_available() and torch.cuda.device_count() > 0 :
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"Rank {rank} using device: {device}", flush=True)
    else:
        device = torch.device("cpu")
        print(f"Rank {rank} using CPU", flush=True)


    # Get the dataframes and subjects based on the classification task
    # This part usually doesn't need distributed handling if done before data splitting per fold
    df, subjects = load_dataframe(config['dataset_csv'], config['task'])

    if is_main_process:
        print(f"Number of subjects: {len(subjects)}", flush=True)

    # Define k-fold cross validation
    kf = KFold(n_splits=config['k_folds'],
               shuffle=True,
               random_state=config['random_seed'])

    # Create lists to store the metrics for each fold (only on main process)
    fold_metrics = []
    all_y_true_list = [] # Store y_true from each fold's main process
    all_y_pred_list = [] # Store y_pred from each fold's main process

    # Perform cross-validation on the dataset based on the subjects
    for fold_idx, (train_val_subj_index, test_subj_index) in enumerate(kf.split(subjects)):
        current_fold = fold_idx + 1
        if is_main_process:
            print(f"\n\n ----------------------- Fold {current_fold} ----------------------- \n\n", flush=True)

        # Split the dataset in train, val and test by subject
        # This split logic is independent of rank and should be consistent
        train_df, val_df, test_df = train_val_test_subject_split(df=df,
                                                                 train_val_subj=subjects[train_val_subj_index],
                                                                 test_subj=subjects[test_subj_index],
                                                                 val_perc_split=config['val_perc_split'],
                                                                 random_seed=config['random_seed'])

        # Set seeds for reproducibility - crucial for DDP to start identically
        # Each process might need a slightly different seed if stochasticity
        # within the process needs to differ, but often setting the same seed
        # before model/optimizer init is enough. Let's use rank for variation.
        set_seeds(config['random_seed'] + rank)

        # Instantiate the tensorboard writer for this fold (only on main process)
        writer = None
        if is_main_process:
            writer = create_writer(
                config=config,
                fold_num=current_fold,
                timestamp=timestamp,
                extra=config["extra"]
            )

        # Run the experiment
        y_true, y_pred, test_metrics = experiments.run_experiment(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                config=config,
                writer=writer, # Writer is None for non-main processes
                device=device, # The device assigned to this rank
                fold=current_fold,
                rank=rank,          # Pass rank
                world_size=world_size # Pass world_size
            )

        # --- Collect Results (Only on Main Process) ---
        # y_true, y_pred, test_metrics are returned ONLY by rank 0 of the run_experiment call
        if is_main_process:
            all_y_true_list.append(y_true)
            all_y_pred_list.append(y_pred)
            fold_metrics.append(test_metrics)
            print(f"Fold {current_fold} Test Metrics (on Rank 0): {test_metrics}", flush=True)


        # Barrier to ensure all processes finish the fold before starting the next
        if world_size > 1:
            dist.barrier()

    # --- Aggregate and Save Results (Only on Main Process) ---
    if is_main_process:
        # Calculate the mean of each metric
        if fold_metrics: # Ensure list is not empty
             mean_metrics = {
                 metric: np.mean([entry[metric] for entry in fold_metrics])
                 for metric in fold_metrics[0] # Assuming all dictionaries have the same keys
             }
             # Determine log directory (needs writer object from last fold if not skipping training)
             log_dir = ""
             if config['skip_training']:
                 log_dir = config['experiment_folder']
             elif writer: # Check if writer exists (it should if !skip_training)
                 log_dir = os.path.join(writer.get_logdir(), "../")
             else: # Fallback if writer somehow wasn't created
                 log_dir = os.path.join(config['experiment_folder'], f"{timestamp}_fold_{config['k_folds']}")
                 os.makedirs(log_dir, exist_ok=True)


             results_path = os.path.join(log_dir, "results.json")
             with open(results_path, "w") as f:
                 json.dump(fold_metrics, f, indent=4)
             print(f"Fold results saved to {results_path}", flush=True)


             print("\n\n------------- Mean metrics (across folds): -------------", end="\n\n", flush=True)
             for metric, mean_value in mean_metrics.items():
                 print(f"Mean {metric}: {mean_value}", flush=True)


             # Compute the overall metrics across all folds
             overall_y_true = torch.cat(all_y_true_list, dim=0)
             overall_y_pred = torch.cat(all_y_pred_list, dim=0)
             overall_metrics = compute_metrics(y_true=overall_y_true, y_pred=overall_y_pred, num_classes=len(config['task']))


             overall_metrics_path = os.path.join(log_dir, "overall_metrics.json")
             with open(overall_metrics_path, "w") as f:
                 json.dump(overall_metrics, f, indent=4)
             print(f"Overall metrics saved to {overall_metrics_path}", flush=True)


             config_path = os.path.join(log_dir, "config.json")
             with open(config_path, "w") as f:
                 json.dump(config, f, indent=4)
             print(f"Config saved to {config_path}", flush=True)


             print("\n\n------------- Overall metrics (all folds combined): -------------", end="\n\n", flush=True)
             for metric, value in overall_metrics.items():
                 print(f"{metric}: {value}", flush=True)


             print("\n\n------------- Metrics for each fold: -------------", end="\n\n", flush=True)
             for i, fold_metric in enumerate(fold_metrics):
                 print(f"Fold {i + 1}: {fold_metric}", flush=True)
        else:
             print("No fold metrics collected (fold_metrics list is empty).", flush=True)

    # --- Distributed Cleanup ---
    cleanup_distributed()

# Add this helper function for broadcasting Python objects
def broadcast_object(obj, rank, device):
    if dist.is_initialized():
        objects = [obj if rank == 0 else None]
        dist.broadcast_object_list(objects, src=0, device=device)
        return objects[0]
    return obj


if __name__ == "__main__":
    # Print the PyTorch version used (all ranks will do this)
    print(f"PyTorch version: {torch.__version__}", flush=True)
    # Run the main function
    main()
