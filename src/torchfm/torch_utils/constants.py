# Constants for string parameters
unknown = 'unknown'
missing = 'missing'
label = 'label'
value2index_json = '.value2index.json'
index2value_json = '.index2value.json'
frequent_values_pkl = '.frequent_values.pkl'

wrapper = 'wrapper'
preprocessed = "preprocessed"
txt = ".txt"
csv = ".csv"
train = "train"
validation = "validation"
test = "test"
val = "val"
fwfm = "fwfm"
fm = 'fm'
lr = 'lr'
dcn = 'dcn'
afm = 'afm'
lowrank_fwfm = "lowrank_fwfm"
pruned_fwfm = "pruned_fwfm"
tensorfm = "tensorfm"
logloss = "logloss"
auc = "auc"
mse = "mse"
sep = ","
minimize = "minimize"
maximize = "maximize"
reg_param = "reg_param"

criteo = "criteo"
avazu = "avazu"
movielens = "movielens"
triple_dataset = "triple-dataset"
random_binary_function = "random_binary_function"
random_binary_function_4_cols = "random_binary_function_4_cols"
compas = "compas"

# Constants for Model parameters
debug_print = False
sparseGrads = True
epochs_num = 5  # 5 full epochs
batch_sizes_to_check = [1024]
num_batches_in_epoch = 100 # 3000  # 100
do_partial_epochs = True
emb_sizes = [8, 16]
# top_k_percent = 0.05
weight_decay = 0
coef_vectors_max = 1e-4  # increase, it was 1e-2, 1e-4
coef_vectors_min = 0.0
coef_biases_max = 1e-4   # increase, it was 1e-2, 1e-4
coef_biases_min = 0.0
lr_min = 1e-5
lr_max = 0.1

models_to_check = [fwfm, lowrank_fwfm, pruned_fwfm]
metrics_to_optimize = [logloss, auc]
ranks_to_check = [1, 2, 3, 4, 5]
device_inds = list(range(4))

# Default TensorFM parameters
dim_int = [0]
ten_ranks = [0]

# Paths
user = "<username>"
base_path_project = f"/home/{user}/persistent_drive/pytorch-tensorfm/data"
tmp_save_dir = '{}/tmp_save_dir'.format(base_path_project)

dataset_name = random_binary_function_4_cols  # triple_dataset # avazu # random_binary_function
test_datasets_path = "{}/test-datasets/{}".format(base_path_project,dataset_name)

optuna_num_trials = 50 # was 50 then 100
debug_info_file = f"{tmp_save_dir}/debug_info.txt"
save_optuna_results_file = f"{tmp_save_dir}/optuna_results.txt"
save_run_results = f"{tmp_save_dir}/run_results/"
optuna_journal_log = f"{tmp_save_dir}/optuna-journal"

optuna_journal_log_fwfm = f"{tmp_save_dir}/optuna-journal_fwfm"
optuna_journal_log_lr_fwfm = f"{tmp_save_dir}/optuna-journal_lr_fwfm"
optuna_journal_log_pruned_fwfm = f"{tmp_save_dir}/optuna_journal_log_pruned_fwfm"

# preprocessing constants
default_base_filename = 'train1M'  #'train'
original_input_file_path = '{}/{}.txt'.format(test_datasets_path, default_base_filename)

# Default global settings
torch_global_seed = 42
python_random_seed = 42
optuna_seed = 42
use_batch_iterator = True

evals_on_same_parameters = 1  # 30
hdfs_run = False