import torch
from torch import nn


class RNABertForRRInter(nn.Module):

    NUM_FILTERS = 320
    KERNEL_SIZE = 12
    PROB_DROPOUT = 0.1
    IN_CH = 5

    def __init__(self, bert,
                 embedding_dim=64):

        super(RNABertForRRInter, self).__init__()

        n_in = embedding_dim
        self.bert = bert

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

    def forward(self, tokens, input_ids):
        x = self.embedder(tokens)
        x = x.permute(0, 2, 1)
        # x = torch.transpose(x, perm=[0, 2, 1])

        encoded_layers, _ = self.bert(input_ids)
        embeddings = encoded_layers[-1].detach()

        print(x.shape)
        print(embeddings.shape)
        exit()
        # embeddings = paddle.transpose(embeddings, perm=[0, 2, 1])
        x = torch.concat([x, embeddings], axis=1)

        x, _ = self.cnn_lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)

        return x
