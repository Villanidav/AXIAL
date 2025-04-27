from sklearn.model_selection import train_test_split


def train_val_test_subject_split(df,
                                 train_val_subj,
                                 test_subj,
                                 val_perc_split,
                                 random_seed):
    """
    Split the dataset in train, val and test by subject
    """
    # Split the train_val subjects
    train_subjects, val_subjects = train_test_split(train_val_subj,
                                                    test_size=val_perc_split,
                                                    random_state=random_seed)
    # Split the dataset in train and test
    train_df = df[df['subject'].isin(train_subjects)]
    val_df = df[df['subject'].isin(val_subjects)]
    test_df = df[df['subject'].isin(test_subj)]
    return train_df, val_df, test_df
