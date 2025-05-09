import warnings
from src.data.dataset import ADNIDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from bids import BIDSLayout
import pandas as pd
import torchvision
import numpy as np
import random
import torch
import os


def visualize_sliced_mri(mri, num_slices=80, title=None):
    """ Visualize a sliced MRI image.
    """
    # Create a figure with the given title
    fig, axs = plt.subplots(int(num_slices / 10), 10, figsize=(20, 20))
    for i, ax in enumerate(axs.flat):
        img = mri[:, :, mri.shape[2] // 2 + i - int(num_slices / 2)]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    # Set the title of the figure
    if title is not None:
        fig.suptitle(title, fontsize=20)
    plt.show()



def get_device(cuda_idx=[0]) -> torch.device:
    """ 
    Return the device to use for training and inference.
    Args: cuda_idx (int or list of int): The cuda device index to use. Defaults to 0. If a list is passed, the first
          element will be used as the main GPU and the others as additional GPUs.
    """
    # Set the device order
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Select available device CPU or GPU or MPS (Apple Silicon Macs)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Check what cuda device will be used
    if device.type == 'cuda':
        print(f"{torch.cuda.device_count()} GPU available")
        # Check if the cuda_idx is a list or an integer (if it is a list it means that we want to use multiple GPUs)
        if len(cuda_idx) > 1:
            device = torch.device(f'cuda:{cuda_idx[0]}')
            print(f"\nMain GPU Selected: {torch.cuda.get_device_name(device)} {cuda_idx[0]}")
            print(
                f"Other GPUs: {[torch.cuda.get_device_name(torch.device(f'cuda:{cuda_idx[i]}')) for i in range(1, len(cuda_idx))]}")
        else:
            device = torch.device(f'cuda:{cuda_idx[0]}')
            print(f"\GPU Selected: {torch.cuda.get_device_name(device)} {cuda_idx[0]}\n")
    return device


def set_seeds(seed: int = 42):
    """ Set the seeds for reproducibility.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set the seed for numpy
    np.random.seed(seed)
    # Set the seed for python random module
    random.seed(seed)
    # Print
    print(f"\nSeeds set to {seed}\n")


def create_dataset_csv_from_bids(bids_path: str):
    """
    Create a csv file with the subject, mri path and diagnosis for each subject in the dataset.
    The csv file will be saved in the same directory as the BIDS dataset with the name dataset.csv
    The csv file can be easily loaded with pandas using pd.read_csv('dataset.csv')
    """
    # Create a BIDSLayout object pointing to the dataset
    layout = BIDSLayout(bids_path)
    # Create dataframe with subject, mri path and diagnosis
    df = pd.DataFrame(columns=['subject', 'mri_path', 'diagnosis'])
    subjects = layout.get_subjects()
    for i, subject in enumerate(subjects):
        # Get the sessions for the subject
        subject_sessions = layout.get_sessions(subject=subject)
        # Get the path of the sessions.tsv file
        subject_session_file = layout.get(subject=subject, suffix='sessions')[0]
        # Get the dataframe with only sessions and diagnosis from the sessions.tsv file
        subject_sessions_df = subject_session_file.get_df()[['session_id', 'diagnosis']]
        # Filter the dataframe to keep only the sessions available in the dataset for the subject
        subject_sessions_df = subject_sessions_df[
            subject_sessions_df['session_id'].isin(['ses-' + item for item in subject_sessions])]
        # Get all the smri images for the subject 
        for session in subject_sessions:
            # Get all the smri images for the subject with skullstripped
            subject_smri_file = \
                layout.get(subject=subject, extension='nii.gz', session=session, return_type='filename')[0]
            # Remove the extension from the filename and add the suffix with skullstripped
            subject_smri_file = os.path.splitext(os.path.splitext(subject_smri_file)[0])[0]
            subject_smri_file = subject_smri_file + '_space-MNI152_desc-preproc-N4-skullstripped.nii.gz'
            # Get the label associated to the session
            session_label = \
                subject_sessions_df[subject_sessions_df['session_id'] == 'ses-' + session]['diagnosis'].values[0]
            # Define the new row to be added
            new_row = {'subject': subject, 'mri_path': subject_smri_file, 'diagnosis': session_label}
            # Use the loc method to add the new row to the DataFrame
            df.loc[len(df)] = new_row
        # Print the progress
        print(f"Processed {i + 1} of {len(subjects)} subjects")
    # Save the dataframe to a csv file
    df.to_csv(os.path.join(bids_path, 'dataset.csv'), index=False)


def create_conversion_dataset_csv_from_bids(bids_path, conversion_time):
    """
    Creates a csv file with the path of the sMRI images and the diagnosis for each subject that has conversion
    from MCI to Dementia in less than a certain number of months. The csv file is saved in the dataset folder.
    Parameters:
    dataset_path (str): Path to the dataset in BIDS format.
    conversion_time (int): Number of months to check for conversion from MCI to Dementia.
    """
    # Ignore warnings
    warnings.filterwarnings('ignore')
    # Create a BIDSLayout object pointing to the dataset
    layout = BIDSLayout(bids_path)
    # Get all the sessions
    sessions = layout.get(suffix='sessions')
    # Create dataframes with the cdr scores, diagnosis and mmse score for each session for subjects that have conversion
    pMCI = pd.DataFrame(columns=['subject', 'cdr_score', 'diagnosis', 'mmse_score'])
    sMCI = pd.DataFrame(columns=['subject', 'cdr_score', 'diagnosis', 'mmse_score'])
    df = pd.DataFrame(columns=['subject', 'mri_path', 'diagnosis'])
    for session in sessions:
        # Get the diagnosis for this session and the session id
        session_df = session.get_df().dropna(subset=['diagnosis'])
        # Get the labels of the sessions with an sMRI image
        session_labels = layout.get_sessions(subject=session.subject)
        # Get the labels of the sessions to check
        session_to_check = [f'{item[:-2]}{int(item[-2:]) + conversion_time:02}' for item in session_labels]
        print("Subject: ", session.subject, " MRI-ses:", session_labels, " ses-to-check:", session_to_check)
        for i, ses_label in enumerate(session_labels):
            # Check if the ses_label is in the session_df
            if session_df[session_df['session_id'] == 'ses-' + ses_label].empty:
                print('No diagnosis for ' + ses_label + ' is present a sMRI image')
            else:
                if session_df[session_df['session_id'] == 'ses-' + ses_label]['diagnosis'].values[0] == 'MCI':
                    # Check if session_to_check[i] is in the session_df
                    if session_df[session_df['session_id'] == 'ses-' + session_to_check[i]].empty:
                        print('No diagnosis at ' + session_to_check[i])
                    else:
                        if session_df[session_df['session_id'] == 'ses-' + session_to_check[i]]['diagnosis'].values[0] == 'AD':
                            new_row = {'subject': session.subject, 'cdr_score': session.get_df()['cdr_global'].values,
                                       'diagnosis': session_df['diagnosis'].values,
                                       'mmse_score': session.get_df()['MMSE'].values}
                            pMCI.loc[len(pMCI)] = new_row
                            # Get the smri image for the subject
                            subject_smri_file = \
                                layout.get(subject=session.subject, extension='nii.gz', session=ses_label,
                                           return_type='filename')[0]
                            # Remove the extension from the filename and add the suffix with skullstripped
                            subject_smri_file = os.path.splitext(os.path.splitext(subject_smri_file)[0])[0]
                            subject_smri_file = subject_smri_file + '_space-MNI152_desc-preproc-N4-skullstripped.nii.gz'
                            # Add to the dataframe a column with the path of the smri image
                            new_row = {'subject': session.subject, 'mri_path': subject_smri_file, 'diagnosis': 'pMCI'}
                            df.loc[len(df)] = new_row
                        elif session_df[session_df['session_id'] == 'ses-' + session_to_check[i]]['diagnosis'].values[0] == 'MCI':
                            new_row = {'subject': session.subject, 'cdr_score': session.get_df()['cdr_global'].values,
                                       'diagnosis': session_df['diagnosis'].values,
                                       'mmse_score': session.get_df()['MMSE'].values}
                            sMCI.loc[len(sMCI)] = new_row
                            # Get the smri image for the subject
                            subject_smri_file = \
                                layout.get(subject=session.subject, extension='nii.gz', session=ses_label,
                                           return_type='filename')[0]
                            # Remove the extension from the filename and add the suffix with skullstripped
                            subject_smri_file = os.path.splitext(os.path.splitext(subject_smri_file)[0])[0]
                            subject_smri_file = subject_smri_file + '_space-MNI152_desc-preproc-N4-skullstripped.nii.gz'
                            # Add to the dataframe a column with the path of the smri image
                            new_row = {'subject': session.subject, 'mri_path': subject_smri_file, 'diagnosis': 'sMCI'}
                            df.loc[len(df)] = new_row
        print('\n')
    # Check the number of MRIs associated to a subject that have conversion from MCI to Dementia and the number of MRIs
    # that don't
    print(f"Number of MRIs associated to a progressive MCI: {len(pMCI)}")
    print(f"Number of MRIs associated to a stable MCI: {len(sMCI)}")
    print(f"Number of images: {len(df)}")
    # Save the dataframe to csv file
    df.to_csv(os.path.join(bids_path, 'dataset_conversion_' + str(conversion_time) + 'months' + '.csv'), index=False)


def plot_loss_curves(results):
    """
    Plots training curves of a results dictionary.
    """
    loss = results["train_loss"]
    val_loss = results["val_loss"]
    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    return plt
