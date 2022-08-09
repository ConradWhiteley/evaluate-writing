import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import make_directory_if_no_exists


def normalise(text):
    """
    Normalise the text.
    - All lowercase.
    - Strip whitespaces
    - Remove end of line characters
    :param text:
    :return:
    """
    text = text.lower()
    text = text.strip()
    text = re.sub("\n", " ", text)
    return text

def process_data(data):
    """
    Process the data.
    :return:
    """
    # normalise text
    data['discourse_text'] = data['discourse_text'].apply(normalise)

    if 'discourse_effectiveness' in data.columns:
        # create labels
        effectiveness_labels = {
            "Adequate": 0,
            "Effective": 1,
            "Ineffective": 2,
        }
        data['label'] = data['discourse_effectiveness'].map(effectiveness_labels)

        # save class labels
        reference_directory = './data/reference'
        make_directory_if_no_exists(reference_directory)
        pd.Series(effectiveness_labels).to_csv(os.path.join(reference_directory, 'label_encodings.csv'))
    return data

def load_data():
    """
    Load train and test data.
    :return:
    """
    processed_directory = './data/processed'
    try:
        train_data = pd.read_csv(os.path.join(processed_directory, 'train.csv'))
        val_data = pd.read_csv(os.path.join(processed_directory, 'val.csv'))
        test_data = pd.read_csv(os.path.join(processed_directory, 'test.csv'))
    except FileNotFoundError:
        # load raw data and process it
        train_data = process_data(pd.read_csv('./data/raw/train.csv'))
        test_data = process_data(pd.read_csv('./data/raw/test.csv'))

        # split train data into validation data, stratify on labels to keep class imbalance
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42,
                                                stratify=train_data['label'])

        # save processed data
        make_directory_if_no_exists(processed_directory)
        train_data.to_csv(os.path.join(processed_directory, 'train.csv'))
        val_data.to_csv(os.path.join(processed_directory, 'val.csv'))
        test_data.to_csv(os.path.join(processed_directory, 'test.csv'))

    return (train_data, val_data, test_data)