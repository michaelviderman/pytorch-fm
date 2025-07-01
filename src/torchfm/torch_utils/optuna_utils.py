from src.torchfm.torch_utils.constants import optuna_journal_log


# def save_to_file(study, model_name, metric_top_optimize, best_params, best_trial):
#     study_stats = get_study_stats(study, best_trial, model_name, metric_top_optimize)
#     save_all_args_to_file(study_stats)
#     # f.write(str(best_params) + "\n" + str(best_trial) + "\n\n")


def create_journal_name(base_journal_name, suffix):
    return base_journal_name + suffix + '.log'


def get_journal_name(model_name, metric_top_optimize, top_k_rank, emb_size, dim_int, ten_ranks):
    model_basic_params = f"{model_name}_{metric_top_optimize}_{top_k_rank}_{emb_size}"
    dim_int_str = "_".join([str(i) for i in dim_int])
    ten_ranks_str = "_".join([str(i) for i in ten_ranks])
    return f"{optuna_journal_log}_{model_basic_params}_{dim_int_str}_{ten_ranks_str}.log"


def erase_content_journal(journal_log):
    file_to_delete = open(journal_log, 'w+')
    file_to_delete.close()


def prune_running_if_needed(trial, valid_err, epoch_i):
    # Handle pruning based on the intermediate value.
    if trial is not None:
        import optuna  # dependency on optuna only if required
        trial.report(valid_err, epoch_i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
