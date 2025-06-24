import torch

from src.torchfm.layer import FeaturesLinear


class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, num_features, is_multivalued=False):
        super().__init__()
        self.linear = FeaturesLinear(num_features, is_multival=is_multivalued)

    def forward(self, x, return_l2=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param return_l2: whether to return the l2 regularization term
        """
        scores, reg = self.linear(x, return_l2)
        return scores.squeeze(1), [0.0, reg]        # torch.sigmoid()    - remove sigmoid since train/test with bcewithlogit
