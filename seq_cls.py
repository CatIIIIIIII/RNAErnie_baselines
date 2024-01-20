import torch
from torch import nn


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size, num_labels):
        super(RNABertForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        _, pooled_output = self.bert(input_ids)
        logits = self.classifier(pooled_output)
        return logits
