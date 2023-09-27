import torch

from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, is_multivalued=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, is_multival=is_multivalued)
        self.linear = FeaturesLinear(field_dims, is_multival=is_multivalued)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1)  #torch.sigmoid()  - remove sigmoid since train/test with bcewithlogit
