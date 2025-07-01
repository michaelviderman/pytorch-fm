import numpy as np
import pandas as pd
import torch

from src.torchfm.torch_utils.io_utils import read_pd_dataframe


class WrapperMultivaluedDataset(torch.utils.data.Dataset):
    """
    General Multivalued Dataset

    Data preparation
        splits the dataset to the features and the label

    :param dataset_path: dataset path
    """
    multivalued = True

    def __init__(self, dataset_path, sep=',', secondary_sep='|', engine='c', header='infer'):
        miltival_col_name = 'genres'
        data = read_pd_dataframe(dataset_path, sep, engine, header)
        self.targets = data['label'].to_numpy().astype(np.float32)
        data = data.drop('label', axis=1)

        data[miltival_col_name] = data[miltival_col_name].str.split(secondary_sep).apply(lambda x: [int(i) for i in x])

        self.max_length = max(data[miltival_col_name].map(len))

        num_columns = len(data.columns)
        global_offsets = list(range(num_columns - 1))

        self.weights = np.vstack(data[miltival_col_name].apply(lambda x: WrapperMultivaluedDataset._create_weights(x, self.max_length, num_columns - 1)).to_numpy()).astype(np.float32)
        self.offsets = np.vstack(data[miltival_col_name].apply(lambda x: WrapperMultivaluedDataset._create_offsets(x, num_columns - 2, global_offsets)).to_numpy()).astype(np.int32)

        data[miltival_col_name] = data[miltival_col_name].apply(lambda x: x + [0] * (self.max_length - len(x)))

        lst_genres = ['genre' + str(i) for i in range(0, 6)]
        df_genres = pd.DataFrame(data[miltival_col_name].to_list(), columns=lst_genres)

        self.indices = pd.concat([data.drop([miltival_col_name], axis=1), df_genres], axis=1, join='inner').to_numpy().astype(np.int32)
        self.field_dims = np.max(self.indices)
        self.num_columns = num_columns

    def __len__(self):
        return self.targets.shape[0]

    # def __getitem__(self, index):
    #     return np.concatenate((self.indices[index], self.offsets[index], self.weights[index]), axis=0, dtype='float32'), self.targets[index]

    @staticmethod
    def _create_weights(lst, full_length, num_cols, align_val=0.0):
        length = len(lst)
        weights = np.empty if length == 0 else np.array([1.0] * num_cols + [1.0 / length] * length + [align_val] * (full_length - length))
        return weights.astype(np.float32)

    @staticmethod
    def _create_offsets(lst, num_cols, global_offsets):
        return np.array(global_offsets + [num_cols + len(lst)])
