
from base_classes import BaseMetrics


class SeqClsMetrics(BaseMetrics):
    """Metrics for classification
    """

    def __call__(self, outputs, labels):
        """call function
        """
        return super().__call__(outputs, labels)
