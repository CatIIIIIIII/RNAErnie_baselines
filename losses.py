from torch import nn
import torch.nn.functional as F


class SeqClsLoss(nn.Module):

    def __init__(self):
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels):
        # convert labels to int64
        loss = F.cross_entropy(outputs, labels)
        return loss


class RRInterLoss(nn.Module):

    def __init__(self):
        super(RRInterLoss, self).__init__()

    def forward(self, outputs, labels):
        loss = F.cross_entropy(outputs, labels)
        return loss
