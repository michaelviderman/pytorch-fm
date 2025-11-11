import multiprocessing as mp
from multiprocessing import Process
from main_optuna import run_all_for_device_ind
from src.torchfm.torch_utils.constants import *


# tensorfm_options = [([2],[4]), ([2],[2]), ([2,3],[4,4]), ([2,3],[2,2])]
# tensorfm for Avazu
tensorfm_options = [([2,3],[8,8]), ([2,3],[16,16])] # [([2,3,4],[16,16,16]), ([2,3,4],[22,22,22])]

tensorfm_options_binary_random = [([2,3],[3,3]), ([3],[3])] # [([2,3,4],[16,16,16]), ([2,3,4],[22,22,22])]

lst_tensorfm_options = [(tensorfm, met_to_opt, 0, emb_size, tensorfm_option[0], tensorfm_option[1]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes for tensorfm_option in tensorfm_options_binary_random]

tensorfm_options_binary_random_4_cols = [([2,3],[3,3]), ([2,3,4],[4,4,4])]
lst_tensorfm_options_random_4_cols = [(tensorfm, met_to_opt, 0, emb_size, tensorfm_option[0], tensorfm_option[1]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes for tensorfm_option in tensorfm_options_binary_random_4_cols]

tensorfm_options_binary_random_4_cols_ = [([2,3,4],[1,1,1])]
lst_tensorfm_options_random_4_cols_ = [(tensorfm, met_to_opt, 0, emb_size, tensorfm_option[0], tensorfm_option[1]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes for tensorfm_option in tensorfm_options_binary_random_4_cols_]


fwfm_options = [(fwfm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
fm_options = [(fm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
lr_options = [(lr, met_to_opt, 0, emb_sizes[0], [0], [0]) for met_to_opt in metrics_to_optimize]
dcn_options = [(dcn, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
afm_options = [(afm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]
hofm_options = [(hofm, met_to_opt, 0, emb_size, [0], [0]) for met_to_opt in metrics_to_optimize for emb_size in emb_sizes]

# 8 processes
# Use: 'tmux attach'   to run session to run the python from
# ctrl B, D   --- to disconnect


if __name__ == '__main__':

    queue = mp.Queue()
    for tpl in (lst_tensorfm_options+fwfm_options):
        queue.put(tpl)

    processes = [Process(target=run_all_for_device_ind, args=(queue, dev_ind), daemon=True) for dev_ind in device_inds]

    [p.start() for p in processes]
    print("Started!!!!")
    [p.join() for p in processes]
    print("Ended!!!!")
