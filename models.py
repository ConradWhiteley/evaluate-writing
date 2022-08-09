from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    """
    BERT classification model.
    """
    def __init__(self, n_classes: int, bert_model_name: str = 'bert-base-cased', dropout=0.5):
        """
        Constructor for the BERT classification model.
        :param n_classes: (int) number of labels
        :param dropout: (float) amount of dropout, for regularisation
        """
        # init parent class
        super(BertClassifier, self).__init__()

        # number of labels
        self.n_classes = n_classes

        # load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # set hidden size for given pre-trained model
        bert_model__hidden_size = {
            'bert-base-cased': 768
        }
        if bert_model_name not in bert_model__hidden_size:
            raise NotImplementedError('BERT model not configured.')
        self.hidden_size = bert_model__hidden_size[bert_model_name]

        # dropout, for regularisation
        self.dropout = nn.Dropout(dropout)
        # linear layer
        self.linear = nn.Linear(self.hidden_size, self.n_classes)
        # non-linear activation function
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        """
        Forward steps
        :param input_id:
        :param mask:
        :return:
        """
        _, output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer