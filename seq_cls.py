import torch
from torch import nn


class RNABertForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=120):
        super(RNABertForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        _, pooled_output = self.bert(input_ids)
        logits = self.classifier(pooled_output)
        return logits


class RNAMsmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=768):
        super(RNAMsmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    def forward(self, input_ids):
        output = self.bert(input_ids)
        representations = output["representations"][0, 0, ...]
        logits = self.classifier(representations)
        return logits


class RNAFmForSeqCls(nn.Module):
    def __init__(self, bert, hidden_size=640):
        super(RNAFmForSeqCls, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(hidden_size, 13)

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=True)

    def forward(self, input_ids):
        output = self.bert(input_ids)
        representations = output["representations"][12]
        logits = self.classifier(representations)
        return logits
