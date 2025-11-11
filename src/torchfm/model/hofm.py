import torch

from src.torchfm.layer import FeaturesLinear, FactorizationMachine, AnovaKernel, FeaturesEmbedding


class HighOrderFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Higher-Order Factorization Machines.

    Reference:
        M Blondel, et al. Higher-Order Factorization Machines, 2016.
    """
    def __init__(self, field_dims, embed_dim, order=3, is_multivalued=False):
        super().__init__()
        if order < 1:
            raise ValueError(f'invalid order: {order}')
        self.order = order
        self.embed_dim = embed_dim
        self.linear = FeaturesLinear(field_dims, is_multival=is_multivalued)
        if order >= 2:
            self.embedding = FeaturesEmbedding(field_dims, embed_dim * (order - 1), is_multival=is_multivalued)
            self.fm = FactorizationMachine(reduce_sum=True)
        if order >= 3:
            self.kernels = torch.nn.ModuleList([
                AnovaKernel(order=i, reduce_sum=True) for i in range(3, order + 1)
            ])

    def forward(self, x, return_l2=False):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :param return_l2: whether to return L2 regularization terms
        """
        # Linear term with regularization
        linear_output, reg_linear = self.linear(x, return_l2)
        y = linear_output.squeeze(1)
        
        reg_embedding = 0.0
        
        if self.order >= 2:
            # Embedding with regularization
            emb, reg_emb = self.embedding(x, return_l2)
            reg_embedding = reg_emb
            
            # Second-order interactions (FM part)
            x_part = emb[:, :, :self.embed_dim]
            y += self.fm(x_part).squeeze(1)
            
            # Higher-order interactions (order >= 3)
            if self.order >= 3:
                for i in range(self.order - 2):
                    x_part = emb[:, :, (i + 1) * self.embed_dim: (i + 2) * self.embed_dim]
                    y += self.kernels[i](x_part).squeeze(1)
        
        if return_l2:
            return y, [reg_embedding, reg_linear]
        else:
            return y, [0.0, 0.0]
