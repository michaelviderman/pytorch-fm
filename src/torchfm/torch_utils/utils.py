import random
import os
import numpy as np
import traceback
import torch
import sys

from src.torchfm.torch_utils.batch_iterator import BatchIter
from src.torchfm.torch_utils.constants import *
from src.torchfm.dataset.wrapper_dataset import WrapperDataset
from src.torchfm.dataset.wrapper_multivalued_dataset import WrapperMultivaluedDataset
from src.torchfm.model.afi import AutomaticFeatureInteractionModel
from src.torchfm.model.afm import AttentionalFactorizationMachineModel
from src.torchfm.model.dcn import DeepCrossNetworkModel
from src.torchfm.model.dfm import DeepFactorizationMachineModel
from src.torchfm.model.ffm import FieldAwareFactorizationMachineModel
from src.torchfm.model.fm import FactorizationMachineModel
from src.torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from src.torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from src.torchfm.model.hofm import HighOrderFactorizationMachineModel
from src.torchfm.model.tensorfm import TensorFactorizationMachineModel
from src.torchfm.model.lr import LogisticRegressionModel
from src.torchfm.model.ncf import NeuralCollaborativeFiltering
from src.torchfm.model.nfm import NeuralFactorizationMachineModel
from src.torchfm.model.pnn import ProductNeuralNetworkModel
from src.torchfm.model.wd import WideAndDeepModel
from src.torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from src.torchfm.model.afn import AdaptiveFactorizationNetwork
from src.torchfm.model.fwfm import FieldWeightedFactorizationMachineModel, PrunedFieldWeightedFactorizationMachineModel
from src.torchfm.model.low_rank_fwfm import LowRankFieldWeightedFactorizationMachineModel
from src.torchfm.torch_utils.io_utils import write_debug_info


def print_msg(*args):
    if debug_print:
        print(args)


def append_to_file(spark, string_to_append, hdfs_path):
    try:
        # Read the existing data from the HDFS file
        existing_data = spark.sparkContext.textFile(hdfs_path)

        # Create an RDD with the string to append
        new_data = spark.sparkContext.parallelize([string_to_append])

        # Union the existing data with the new data
        combined_data = existing_data.union(new_data)

        # Write the combined data back to the HDFS file with overwrite=False
        combined_data.saveAsTextFile(hdfs_path)

        print(f"'{string_to_append}' appended to '{hdfs_path}'")

    except Exception as e:
        print(f"Error appending to '{hdfs_path}': {str(e)}")


def get_optimizer(opt_name, parameters, learning_rate, weight_decay=0):
    if opt_name == "adam":
        return torch.optim.Adam(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "sparseadam":
        return torch.optim.SparseAdam(params=parameters, lr=learning_rate)
    elif opt_name == "adagrad":
        return torch.optim.Adagrad(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(params=parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError('unknown optimizer name: ' + opt_name)


def get_criterion(criterion):
    if criterion == 'bceloss':
        return torch.nn.BCELoss()
    elif criterion == 'bcelogitloss':
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "nllloss":
        return torch.nn.NLLLoss()
    elif criterion == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError('unknown criterion name: ' + criterion)


def get_dataset(name, path):
    if movielens in name:
        return WrapperMultivaluedDataset(path)
    elif name in [criteo, 'wrapper']:
        return WrapperDataset(path)
    else:
        return WrapperDataset(path, sep=',')


def get_datasets(dataset_name, dataset_paths):
    train_dataset = get_dataset(dataset_name, dataset_paths[0])
    valid_dataset = get_dataset(dataset_name, dataset_paths[1])
    test_dataset = get_dataset(dataset_name, dataset_paths[2])
    return train_dataset, valid_dataset, test_dataset


def get_iterator(dataset, batch_size, num_workers, device, shuffle):
    return BatchIter(dataset, device, batch_size, shuffle)
    # DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, generator=get_seeded_generator())


def get_iterators(train_dataset, valid_dataset, test_dataset, batch_size, num_workers, device):
    train_data_loader = get_iterator(train_dataset, batch_size, num_workers, device, shuffle=True)
    valid_data_loader = get_iterator(valid_dataset, batch_size, num_workers, device, shuffle=False)
    test_data_loader = get_iterator(test_dataset, batch_size, num_workers, device, shuffle=False)

    train_data_loader_loss = get_iterator(train_dataset, batch_size, num_workers, device, shuffle=False)
    return train_data_loader, valid_data_loader, test_data_loader, train_data_loader_loss


def get_model(name, dataset, rank_param, emb_size, tensor_fm_params=None):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    num_features = dataset.field_dims
    num_columns = dataset.num_columns
    is_multival = dataset.multivalued

    if name == 'lr':
        return LogisticRegressionModel(num_features=num_features, is_multivalued=is_multival)
    elif name == 'fm':
        return FactorizationMachineModel(num_features, embed_dim=emb_size, is_multivalued=is_multival)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(num_features, embed_dim=emb_size, order=3, is_multivalued=is_multival)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(num_features, embed_dim=4)
    elif name == 'fwfm':
        return FieldWeightedFactorizationMachineModel(num_features=num_features, embed_dim=emb_size,
                                                      num_fields=num_columns, is_multivalued=is_multival)
    elif name == 'pruned_fwfm':
        topk = rank_param * (num_columns + 1)
        return PrunedFieldWeightedFactorizationMachineModel(num_features=num_features, embed_dim=emb_size,
                                                            num_fields=num_columns, topk=topk,
                                                            is_multivalued=is_multival)
    elif name == 'lowrank_fwfm':
        return LowRankFieldWeightedFactorizationMachineModel(num_features=num_features, embed_dim=emb_size,
                                                             num_fields=num_columns, c=rank_param,
                                                             is_multivalued=is_multival)
    # dim_int = [d_1,...,d_l] and rank_tensors = [r_1,...,r_l] are two lists
    # For an index 1 <= i <= l, we consider a d_i-order interaction of rank r_i
    elif name == 'tensorfm':
        return TensorFactorizationMachineModel(num_features, num_columns, emb_size, dim_int=tensor_fm_params.dim_int,
                                               rank_tensors=tensor_fm_params.ten_ranks, is_multivalued=is_multival)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(num_features, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(num_features, num_columns, embed_dim=emb_size, num_layers=2, mlp_dims=(16, 16), dropout=0.2, is_multival=is_multival)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(num_features, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    # elif name == 'ncf':
    #     # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
    #     assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
    #     return NeuralCollaborativeFiltering(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
    #                                         user_field_idx=dataset.user_field_idx,
    #                                         item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(num_features, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(num_features, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            num_features, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(num_features, embed_dim=emb_size, attn_size=8, dropouts=(0.2, 0.2), is_multival=is_multival)
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            num_features, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400),
            dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            num_features, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


def get_baselines_log_loss(targets):
    log_loss = torch.nn.BCELoss()
    targets_ctr = torch.sum(targets) / targets.size(dim=0)
    ctr_loss = log_loss(torch.ones_like(targets) * targets_ctr.item(),
                        targets).item()  # global train ctr 0.22711533894173677
    half_loss = log_loss((torch.ones_like(targets) * 0.5).float(), targets).item()

    return ctr_loss, half_loss


def get_absolute_sizes(total_len, rel_sizes):
    assert np.sum(rel_sizes) == 1.0
    absolute_sizes = [round(rel * total_len) for rel in rel_sizes[:-1]]
    absolute_sizes.append(total_len - np.sum(absolute_sizes))
    assert total_len == np.sum(absolute_sizes)
    return absolute_sizes


class EarlyStopper(object):
    def __init__(self, tolerance=2, min_delta=0.05):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
        else:
            if self.best_loss + self.min_delta < loss:
                self.counter += 1
                if self.counter >= self.tolerance:
                    self.early_stop = True


class LossCalc:
    total_loss = 0
    total_ctr_loss = 0  # loss w.r.t. ctr prediction on data
    total_half_loss = 0  # loss w.r.t. constant 1/2 prediction

    def __init__(self, total_loss, total_ctr_loss, total_half_loss):
        self.total_loss = total_loss
        self.total_ctr_loss = total_ctr_loss
        self.total_half_loss = total_half_loss

    def add(self, total_loss, total_ctr_loss, total_half_loss):
        self.total_loss += total_loss
        self.total_ctr_loss += total_ctr_loss
        self.total_half_loss += total_half_loss

    def remove_results(self):
        self.total_loss = 0
        self.total_ctr_loss = 0
        self.total_half_loss = 0


class BestError:
    best_auc = 0.0
    best_logloss = sys.float_info.max

    def __init__(self):
        pass

    def update(self, tmp_logloss, tmp_auc):
        self.best_auc = tmp_auc if tmp_auc > self.best_auc else self.best_auc
        self.best_logloss = tmp_logloss if tmp_logloss < self.best_logloss else self.best_logloss


def get_from_queue(q):
    try:
        return q.get(timeout=5.0)
    except Exception as e:
        write_debug_info("get_from_queue cannot get item", str(e), traceback.format_exc())
        return


class EpochStopper(object):
    def __init__(self, num_batches_in_epoch, do_partial_epochs):
        self.num_batches_in_epoch = num_batches_in_epoch
        self.do_partial_epochs = do_partial_epochs
        self.epoch_stop = False
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.do_partial_epochs:
            self.epoch_stop = (self.counter >= self.num_batches_in_epoch)

    def restart(self):
        self.epoch_stop = False
        self.counter = 0


def set_torch_seed(tmp_seed=0):
    torch_seed = torch_global_seed + tmp_seed
    python_seed = python_random_seed + tmp_seed

    torch.manual_seed(torch_seed)
    # torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    random.seed(python_seed)
    np.random.seed(python_seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(python_seed)


def get_seeded_generator():
    g = torch.Generator()
    g.manual_seed(torch_global_seed)
    return g


def get_device_str(device_ind):
    return ('cuda' if torch.cuda.is_available() else 'cpu') + ":" + str(device_ind)
