import traceback
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from main import top_main_for_option_run
from src.torchfm.torch_utils.options_to_run import Option2Run, TensorFMParams
from src.torchfm.torch_utils.constants import optuna_num_trials, logloss, mse, minimize, maximize, batch_sizes_to_check, \
    coef_vectors_max, coef_biases_max, coef_vectors_min, coef_biases_min, optuna_seed, lr_min, lr_max
from src.torchfm.torch_utils.optuna_utils import get_journal_name, erase_content_journal
from src.torchfm.torch_utils.utils import get_from_queue, set_torch_seed
from src.torchfm.torch_utils.io_utils import write_debug_info

def objective(study, trial, model_name, device_ind, metric_to_optimize, rank_param, emb_size, dim_int, ten_ranks):
    lr = trial.suggest_float('lr', lr_min, lr_max, log=True)
    opt_name = "adagrad"  # other optimizers causing issues with sparse/dense gradients
    batch_size = trial.suggest_categorical("batch_size", batch_sizes_to_check)

    coef_vectors = trial.suggest_float("coef_vectors", coef_vectors_min, coef_vectors_max)  # reg coef vectors
    coef_biases = trial.suggest_float("coef_biases", coef_biases_min, coef_biases_max)  # reg coef biases

    option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, lr, opt_name, batch_size, TensorFMParams(dim_int, ten_ranks), coef_vectors, coef_biases, 0.0)
    return top_main_for_option_run(study, trial, device_ind, option_to_run)


def run_optuna_study(model_name, metric_to_optimize, rank_param, emb_size, device_ind, dim_int, ten_ranks):

    journal_name = get_journal_name(model_name, metric_to_optimize, rank_param, emb_size, dim_int, ten_ranks)
    erase_content_journal(journal_name)
    storage = JournalStorage(JournalFileStorage(journal_name))
    study = optuna.create_study(study_name=f"Study {model_name} {metric_to_optimize} {rank_param} {emb_size}",
                                storage=storage,
                                direction=(minimize if metric_to_optimize in [logloss, mse] else maximize),
                                pruner=optuna.pruners.NopPruner(),
                                sampler=optuna.samplers.TPESampler(seed=optuna_seed))

    study.optimize(lambda trial: objective(study, trial, model_name, device_ind, metric_to_optimize, rank_param, emb_size, dim_int, ten_ranks), n_trials=optuna_num_trials)


def run_all_for_device_ind(queue, device_ind):
    try:
        set_torch_seed()

        while True:
            elm = get_from_queue(queue)
            if elm is None:
                write_debug_info("run_all_for_device_ind no elm returned. Exit. Device ind: ", str(device_ind))
                return

            model_name, metric_top_optimize, rank_param, emb_size, dim_int, ten_ranks = elm[0], elm[1], elm[2], elm[3], elm[4], elm[5]
            run_optuna_study(model_name, metric_top_optimize, rank_param, emb_size, device_ind, dim_int, ten_ranks)
    except Exception as e:
        write_debug_info("Exception in run_all_for_device_ind: ", str(e), traceback.format_exc())
