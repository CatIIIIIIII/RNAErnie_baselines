from torch import nn
import torch.nn.functional as F


class SeqClsLoss(nn.Module):
    """Loss of sequence classification.
    """

    def __init__(self):
        """
        Returns:
            None
        """
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels):
        """forward function

        Args:
            outputs: [B, C] logit scores
            labels: [N] labels

        Returns:
            Tensor: final loss.
        """
        # convert labels to int64
        loss = F.cross_entropy(outputs, labels)
        return loss
