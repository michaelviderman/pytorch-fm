import torch

from src.torchfm.layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine


class AttentionalFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.

    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts, is_multival=False):
        super().__init__()
        # self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)

    def forward(self, x, return_l2=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param return_l2: whether to return the l2 regularization term
        """
        emb, emb_reg = self.embedding(x, return_l2)
        afm, afm_reg = self.afm(emb, return_l2)
        lin, lin_reg = self.linear(x, return_l2)
        x = afm+lin
        # x = self.linear(x) + self.afm(self.embedding(x))
        return x.squeeze(1), [emb_reg, afm_reg+lin_reg]   # torch.sigmoid()    - remove sigmoid since train/test with bcewithlogit
