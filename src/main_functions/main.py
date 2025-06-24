from torchmetrics.classification import auroc

from src.torchfm.torch_utils.constants import epochs_num, test_datasets_path, num_batches_in_epoch, auc, movielens, mse, \
    dataset_name, do_partial_epochs, evals_on_same_parameters
from src.torchfm.torch_utils.io_utils import get_train_validation_test_preprocessed_paths, save_all_args_to_file
from src.torchfm.torch_utils.optuna_utils import prune_running_if_needed
import torch
import time

from src.torchfm.torch_utils.utils import set_torch_seed, get_device_str, get_iterators, get_datasets, BestError, \
    get_criterion, get_optimizer, EarlyStopper, get_model, print_msg, EpochStopper


def train_batch_step(model, optimizer, criterion, fields, target, option_to_run):
    y, reg = model(fields, option_to_run.return_l2)  # return y,regularization_term_arr
    loss = criterion(y, target)
    total_reg = reg[0] * option_to_run.reg_coef_vectors + reg[1] * option_to_run.reg_coef_biases
    cost = loss + total_reg
    model.zero_grad()  # optimizer.zero_grad()
    cost.backward()
    optimizer.step()


# def train_meta_epoch(model, optimizer, batch_iterator, criterion, device, option_to_run):
#     model.train()
#     epoch_stopper = EpochStopper(num_batches_in_epoch=num_batches_in_epoch, do_partial_epochs=do_partial_epochs)
#     for fields, target in batch_iterator:
#         train_batch_step(model, optimizer, criterion, fields, target, option_to_run)
#         epoch_stopper()
#         if epoch_stopper.epoch_stop:
#             break


# def train_wrapper(model, optimizer, iterator, criterion, device, option_to_run):
#     start = time.time()
#     train_meta_epoch(model, optimizer, iterator, criterion, device, option_to_run)
#     end = time.time()
#     train_time = end - start
#     return train_time


def test(model, batch_iterator, criterion, device):
    model.eval()
    test_loss_sum = torch.zeros(1, device=device)
    test_set_size = 0

    ys = []
    targets = []
    auc = auroc.AUROC(task="binary")  # BinaryAUROC()

    with torch.no_grad():
        for fields, target in batch_iterator:  # tqdm.tqdm(..., smoothing=0, mininterval=1.0):
            y, _ = model(fields)
            test_loss_sum += criterion(y, target) * target.shape[0]
            test_set_size += target.shape[0]
            ys.append(y)
            targets.append(target.type(torch.IntTensor))

    loss = test_loss_sum.item() / test_set_size
    auc_res = auc(torch.cat(ys).to(device), torch.cat(targets).to(device)).item()  # auc.compute().item()

    return loss, auc_res  # loss can be logloss or mse


def valid_test(model, train_iterator_loss, valid_iterator, test_iterator, criterion, device):
    start = time.time()
    valid_err, valid_auc = test(model, valid_iterator, criterion, device)  # , ctr_err, half_err
    end = time.time()
    test_err, test_auc = test(model, test_iterator, criterion, device)

    if train_iterator_loss is not None:
        train_err, train_auc = test(model, train_iterator_loss, criterion, device)
    else:
        train_err, train_auc = None, None

    return train_err, train_auc, valid_err, valid_auc, test_err, test_auc, end - start


def valid_test_save(model, train_iterator_loss, valid_iterator, test_iterator, criterion, device, option_to_run,
                    study_name, trial_number, epoch_i, train_time, criterion_name, dataset_nm, best_error):
    train_err, train_auc, valid_err, valid_auc, test_err, test_auc, valid_time = valid_test(model,
                                                                                            train_iterator_loss,
                                                                                            valid_iterator,
                                                                                            test_iterator,
                                                                                            criterion, device)
    save_all_args_to_file(option_to_run, study_name, option_to_run.to_csv_res(), trial_number, epoch_i, train_err,
                          train_auc, valid_err, valid_auc, test_err, test_auc, train_time, valid_time,
                          criterion_name, dataset_nm)

    best_error.update(valid_err, valid_auc)


def main(dataset_nm, dataset_paths, option_to_run, epoch, criterion_name, weight_decay, device, study=None, trial=None):
    num_workers = 0
    device = torch.device(device)
    study_name = study.study_name if study is not None else ""
    trial_number = trial.number if trial is not None else 0

    train_dataset, valid_dataset, test_dataset = get_datasets(dataset_nm, dataset_paths)
    train_iterator, valid_iterator, test_iterator, train_iterator_loss = get_iterators(train_dataset, valid_dataset,
                                                                                       test_dataset,
                                                                                       option_to_run.batch_size,
                                                                                       num_workers, device)

    model = get_model(option_to_run.m_to_check, train_dataset, option_to_run.rank, option_to_run.emb_size,
                      option_to_run.tensor_fm_params).to(device)
    criterion = get_criterion(criterion_name)
    optimizer = get_optimizer(option_to_run.opt_name, model.parameters(), option_to_run.lr, weight_decay)
    # early_stopper = EarlyStopper()
    best_error = BestError()
    epoch_i = -1

    for epoch_ in range(epoch):
        start = time.time()
        model.train()
        epoch_i += 1
        epoch_stopper = EpochStopper(num_batches_in_epoch=num_batches_in_epoch, do_partial_epochs=do_partial_epochs)

        for fields, target in train_iterator:
            train_batch_step(model, optimizer, criterion, fields, target, option_to_run)
            epoch_stopper()
            if epoch_stopper.epoch_stop:
                epoch_stopper.restart()
                train_time = time.time() - start
                valid_test_save(model, train_iterator_loss, valid_iterator, test_iterator, criterion, device,
                                option_to_run, study_name, trial_number, epoch_i, train_time, criterion_name,
                                dataset_nm, best_error)

                start = time.time()
                model.train()
                epoch_i += 1

        if epoch_stopper.counter > 0:  # The remainder - last mini-epoch in the current pass over the training data
            train_time = time.time() - start
            valid_test_save(model, train_iterator_loss, valid_iterator, test_iterator, criterion, device, option_to_run,
                            study_name, trial_number, epoch_i, train_time, criterion_name, dataset_nm, best_error)

    train_err, train_auc, valid_err, valid_auc, test_err, test_auc, _ = valid_test(model, None, valid_iterator,
                                                                                   test_iterator, criterion, device)
    print_msg(f'valid error: {valid_err} test error: {test_err}')

    return best_error.best_logloss, best_error.best_auc, model


def top_main_for_option_run(study, trial, device_ind, option_to_run):
    import traceback
    set_torch_seed()
    device_str = get_device_str(device_ind)

    train_valid_test_paths = get_train_validation_test_preprocessed_paths(test_datasets_path)

    criterion_name = mse if movielens in train_valid_test_paths[0] else 'bcelogitloss'

    valid_err, valid_auc, model = main(dataset_name, train_valid_test_paths, option_to_run, epochs_num, criterion_name,
                                       0,
                                       device_str, study, trial)

    # to make stats (std, mean) of auc, logloss metrics more reliable we run same training several times
    #     for tmp_ind in range(evals_on_same_parameters-1):
    #         set_torch_seed(tmp_ind+1)
    #         main(dataset_name, train_valid_test_paths, option_to_run, epochs_num, criterion_name, 0,
    #                                     device_str, study, trial)

    return (valid_err if option_to_run.met_to_opt != auc else valid_auc)
