import torch
import torch.nn as nn
from src.torchfm.layer import FeaturesEmbedding, FeaturesLinear
from functools import reduce

class TensorFactorizationMachineModel(torch.nn.Module):
    # dim_int = [d_1,...,d_l] and rank_tensors = [r_1,...,r_l] are two lists
    # For an index 1 <= i <= l, we consider a d_i-order interaction of rank r_i
    def __init__(self, features_dim, num_fields, embed_dim, dim_int, rank_tensors, is_multivalued=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(features_dim, embed_dim, is_multival=is_multivalued)
        self.linear = FeaturesLinear(features_dim, is_multival=is_multivalued)
        self.embed_dim = embed_dim
        self.num_fields = num_fields
        self.dim_int = dim_int
        self.l = len(dim_int)
        self.rank_tensors = rank_tensors

        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(dim_int[i], rank_tensors[i], self.num_fields))
            for i in range(self.l)
        ])

        with torch.no_grad():
            for i in range(self.l):
                nn.init.trunc_normal_(self.W[i], std=0.01)

    def forward(self, x, return_l2=True):  # A = (batch_size, num_fields, embedding_dim)
        emb, reg_emb = self.embedding(x, return_l2)
        ret, reg_linear = self.linear(x, return_l2)

        tmp_tensors_lst = [self.calc_cross(emb, i) for i in range(self.l)]
        ret = reduce(lambda t1, t2: torch.add(t1, t2), tmp_tensors_lst)

        ret = ret.squeeze(-1)

        if return_l2:
            return ret, [reg_emb + self.get_l2_reg(), reg_linear]
        else:
            return ret, [0.0, 0.0]


    def calc_cross(self, A, idx):
        W_i = self.W[idx]
        S = (W_i.unsqueeze(0) @ A.unsqueeze(1))
        S_final = S.prod(dim=1)  
        return torch.sum(S_final, axis=[1,2] ).unsqueeze(1)



    def calc_cross_old(self, A, idx):  # efficiently
        W_i = self.W[idx]  # W_i has shape (d, r, n)
        W_i = W_i.unsqueeze(0)  # Now W_i has shape (1, d, r, n)
        # print("W_i,size", W_i.size())
        # print("A,size",A.size() )
        dot_products = torch.einsum('bljk,bki->bijl', W_i, A)  # (b, emb_dim, r, d)
        prod_over_d = torch.prod(dot_products, axis=3)
        sum_all = torch.sum(prod_over_d, axis=[1, 2])
        return sum_all.unsqueeze(1)

    def get_l2_reg(self):
        tensor_reg = 0
        for i in range(self.l):
            tensor_reg = tensor_reg + torch.sum(self.W[i].square().mean())
        return tensor_reg
