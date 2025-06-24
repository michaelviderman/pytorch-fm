import numpy as np
import torch

from src.torchfm.torch_utils.constants import label
from src.torchfm.torch_utils.io_utils import read_pd_dataframe


class WrapperDataset(torch.utils.data.Dataset):
    """
    General Dataset wrapper

    Data preparation
        splits the dataset to the features and the label

    :param dataset_path: dataset path
    """
    multivalued = False

    def __init__(self, dataset_path, sep='\t', engine='c', header='infer'):
        data = read_pd_dataframe(dataset_path, sep, engine, header)

        self.items = data.loc[:, data.columns != label].to_numpy().astype(np.int32)
        self.targets = data.loc[:, label].to_numpy().astype(np.float32)
        self.field_dims = np.max(self.items)
        self.num_columns = len(self.items[0])

    def __len__(self):
        return self.targets.shape[0]
