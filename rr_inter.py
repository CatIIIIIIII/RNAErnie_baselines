from abc import abstractmethod
import torch
from torch import nn


class BaselinesForRRInter(nn.Module):

    NUM_FILTERS = 320
    KERNEL_SIZE = 12
    PROB_DROPOUT = 0.1
    IN_CH = 5

    def __init__(self, bert, bert_embed_dim,
                 embedding_dim=64):
        super(BaselinesForRRInter, self).__init__()
        self.bert = bert
        n_in = embedding_dim + bert_embed_dim
        self.embedder = nn.Embedding(
            num_embeddings=5, embedding_dim=embedding_dim)
        self.cnn_lstm = nn.Sequential(
            # num_embeddings=5 means "ATCGN"
            nn.Conv1d(in_channels=n_in, out_channels=self.NUM_FILTERS,
                      kernel_size=self.KERNEL_SIZE),
            nn.ReLU(),
            nn.Dropout(self.PROB_DROPOUT),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(self.PROB_DROPOUT),
            nn.LSTM(input_size=27, hidden_size=32, bidirectional=True),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.PROB_DROPOUT),
            nn.Linear(in_features=64, out_features=16),
            nn.Dropout(self.PROB_DROPOUT),
            nn.Linear(in_features=16, out_features=2),
        )

    def _load_pretrained_bert(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)

    @abstractmethod()
    def _extract_embedding(self, input_ids):
        pass

    def forward(self, tokens, input_ids):
        x = self.embedder(tokens)
        x = x.permute(0, 2, 1)
        # x = torch.transpose(x, perm=[0, 2, 1])

        embeddings = self._extract_embedding(input_ids)
        x = torch.concat([x, embeddings], axis=1)
        x, _ = self.cnn_lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


class RNABertForRRInter(BaselinesForRRInter):

    def __init__(self, bert, bert_embed_dim=120):
        super(RNABertForRRInter, self).__init__(bert)

    def _extract_embedding(self, input_ids):
        encoded_layers, _ = self.bert(input_ids)
        embeddings = encoded_layers[-1].detach()
        embeddings = embeddings.permute(0, 2, 1)
        return embeddings


class RNAMsmForRRInter(BaselinesForRRInter):

    def __init__(self, bert, bert_embed_dim=768):
        super(RNAMsmForRRInter, self).__init__(bert, bert_embed_dim)

    def _extract_embedding(self, input_ids):
        output = self.bert(input_ids)
        embeddings = output["representations"][0, 0, ...].detach()
        embeddings = embeddings.permute(0, 2, 1)
        return embeddings
