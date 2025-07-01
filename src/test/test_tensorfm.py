import unittest
import torch

from src.torchfm.model.tensorfm import TensorFactorizationMachineModel


class TestTensorFactorizationMachineModel(unittest.TestCase):

    def test_interaction_matrix_calc(self):
        # dim_int = [d_1,...,d_l] and rank_tensors = [r_1,...,r_l] are two lists
        # For an index 1 <= i <= l, we consider a d_i-order interaction of rank r_i
        embed_dim = 6
        num_features = 20  # as in the fm implemented in this repo
        num_fields = 4
        dim_int = [2]
        rank_tensors = [3]
        model = TensorFactorizationMachineModel(num_features, num_fields, embed_dim, dim_int, rank_tensors)
        with torch.no_grad():
            x = torch.tensor([[[3, 7, 11, 16], [3, 8, 12, 17],[3, 7, 11, 16],[3, 7, 11, 16],[3, 7, 11, 16]]])
            res, reg = model(x)


            # diffTensor = torch.flatten(res1) - torch.flatten(res2)
            # diffTensorNorm = torch.linalg.vector_norm(diffTensor, ord=2)
            # assert diffTensorNorm < 1e-9

    def test_different_implementations(self):
        embed_dim = 6
        num_features = 20  # as in the fm implemented in this repo
        num_fields = 4
        dim_int = [2]
        rank_tensors = [3]
        model = TensorFactorizationMachineModel(num_features, num_fields, embed_dim, dim_int, rank_tensors)
        with torch.no_grad():
            x = torch.tensor([[[3, 7, 11, 16], [3, 8, 12, 17], [3, 7, 11, 16], [3, 7, 11, 16], [3, 7, 11, 16]]])
            emb, reg_emb = model.embedding(x, True)
            res1 = model.calc_cross(emb, 0)
            res2 = model.calc_cross_old(emb, 0)
            print("")
            print(res1)
            print(res2)
            assert torch.allclose(res1, res2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
