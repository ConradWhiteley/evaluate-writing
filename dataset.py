import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class TextClassificationDataset(Dataset):
    """
    Dataset for text classification models.
    """

    def __init__(self, df):
        """
        Constructor for the
        :param df:
        """
        self.labels = df['label']
        self.texts = [
            tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")
            for text in df['discourse_text']
        ]

    def classes(self):
        """
        Return the classes.
        :return:
        """
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        """
        Fetch a batch of labels.
        :param idx:
        :return:
        """
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        """
        Fetch a batch of texts.
        :param idx:
        :return:
        """
        return self.texts[idx]

    def __getitem__(self, idx):
        """
        Fetch an item: text and label
        :param idx:
        :return:
        """
        batch_texts = self.get_batch_texts(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_texts, batch_labels

