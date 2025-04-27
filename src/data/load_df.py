import pandas as pd


def load_dataframe(csv_path, diagnosis):
    """
    Load the dataframe from a csv file and select only the subjects that have a diagnosis based on the task.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    # Get only the MRI paths and the diagnosis with diagnosis CN and AD
    df = df[df['diagnosis'].isin(diagnosis)][['subject', 'mri_path', 'diagnosis']]
    # Get the df subjects
    subjects = df['subject'].unique()
    return df, subjects
