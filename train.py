from datetime import datetime
import json
import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from src.data import train_val_test_subject_split
from src.model import compute_metrics
from src.logger import create_writer
from src.data import load_dataframe
from src.config import load_config
from src.utils import get_device
from src.utils import set_seeds
from src import experiments
import torch.distributed as dist



def main():
    #Get the timestamp for the experiment
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # Load the yaml config file
    config = load_config('config.yaml')
    # Get the device to use
    device = get_device(cuda_idx=config['cuda_device'])
    # Get the dataframes and subjects based on the classification task
    df, subjects = load_dataframe(config['dataset_csv'], config['task'])


    # Define k-fold cross validation
    kf = KFold(n_splits=config['k_folds'],
               shuffle=True,
               random_state=config['random_seed'])

    # Create lists to store the metrics for each fold (only on main process)
    fold_metrics = []
    all_y_true_list = [] # Store y_true from each fold's main process
    all_y_pred_list = [] # Store y_pred from each fold's main process

# Perform cross-validation on the dataset based on the subjects
    for train_val_subj_index, test_subj_index in kf.split(subjects):
        print(f"\n\n ----------------------- Fold {len(fold_metrics) + 1} ----------------------- \n\n")
        # Split the dataset in train, val and test by subject
        train_df, val_df, test_df = train_val_test_subject_split(df=df,
                                                                 train_val_subj=subjects[train_val_subj_index],
                                                                 test_subj=subjects[test_subj_index],
                                                                 val_perc_split=config['val_perc_split'],
                                                                 random_seed=config['random_seed'])
        # Set seeds for reproducibility
        set_seeds(config['random_seed'])
        # Instantiate the tensorboard writer for this fold
        writer = create_writer(
            config=config,
            fold_num=len(fold_metrics) + 1,
            timestamp=timestamp,
            extra=config["extra"]
        )

        y_true, y_pred, test_metrics = experiments.run_experiment(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config,
            writer=writer,
            device=device,
            fold=len(fold_metrics) + 1
        )
        # Concatenate the true and predicted labels for each fold
        all_y_true = torch.cat((all_y_true, y_true), dim=0)
        all_y_pred = torch.cat((all_y_pred, y_pred), dim=0)
        fold_metrics.append(test_metrics)

    # Calculate the mean of each metric
    mean_metrics = {
        metric: np.mean([entry[metric] for entry in fold_metrics])
        for metric in fold_metrics[0]  # Assuming all dictionaries have the same keys
    }

    # Determine log directories
    log_dir = os.path.join(writer.get_logdir(), "../")
    results_path = os.path.join(log_dir, "results.json")
    overall_metrics_path = os.path.join(log_dir, "overall_metrics.json")
    config_path = os.path.join(log_dir, "config.json")

    # Save fold results
    with open(results_path, "w") as f:
        json.dump(fold_metrics, f, indent=4)
    print(f"Fold results saved to {results_path}", flush=True)

    print_metrics(mean_metrics, "Mean")

    # Compute the overall metrics across all folds
    overall_y_true = torch.cat(all_y_true_list, dim=0)
    overall_y_pred = torch.cat(all_y_pred_list, dim=0)
    overall_metrics = compute_metrics(y_true=overall_y_true, y_pred=overall_y_pred, num_classes=len(config['task']))

    # Save the overall metrics
    with open(overall_metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=4)
    print(f"Overall metrics saved to {overall_metrics_path}", flush=True)

    # Save the config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}", flush=True)

    print_metrics(overall_metrics, "Overall")

    # Print the metrics for each fold
    print("\n\n------------- Metrics for each fold: -------------", end="\n\n", flush=True)
    for i, fold_metric in enumerate(fold_metrics):
        print(f"Fold {i + 1}: {fold_metric}", flush=True)


def print_metrics(metrics, metric_type):
    print("\n\n------------- {metric_type} metrics (across folds): -------------", end="\n\n", flush=True)
    for metric, mean_value in metrics.items():
        print(f"{metric_type} {metric}: {mean_value}", flush=True)


if __name__ == "__main__":
    main()
