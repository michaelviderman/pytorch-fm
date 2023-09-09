unknown = 'unknown'
missing = 'missing'
label = 'label'
value2index_json = '.value2index.json'
index2value_json = '.index2value.json'
frequent_values_pkl = '.frequent_values.pkl'

wrapper = 'wrapper'
preprocessed = "preprocessed"
txt = ".txt"
train = "train"
validation = "validation"
test = "test"

debug_print = True
sparseGrads = True
epochs_num = 5
batch_size = 100
top_k_percent = 0.05
weight_decay = 0

base_path_project = ".."
path_torchfm = "{}/torchfm".format(base_path_project)
test_datasets_path = "{}/test-datasets".format(path_torchfm)
tmp_save_dir = '{}/tmp_save_dir'.format(base_path_project)

default_base_filename = 'train1M'

original_input_file_path = '{}/{}.txt'.format(test_datasets_path, default_base_filename)

save_optuna_results_file = f"{tmp_save_dir}/optuna_results.txt"
optuna_journal_log = f"{tmp_save_dir}/optuna-journal.log"
# data_file = "../persistent_drive/data/train100K.txt"
