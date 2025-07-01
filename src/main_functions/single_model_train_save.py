from src.main_functions.main import top_main_for_option_run
from src.torchfm.torch_utils.constants import tensorfm, tmp_save_dir
from src.torchfm.torch_utils.options_to_run import Option2Run, TensorFMParams
import torch


# def run_save_single_model():
#     opt_name = "adagrad"
#     batch_size = 256
#     learning_rate = 0.01578378074
#     model_name = "lowrank_fwfm"
#     device_ind = 0
#     metric_to_optimize = "logloss"
#     emb_size = 8
#     rank_param = 5
#     study = None
#     trial = None
#     reg_coef_vectors = 0.01
#     reg_coef_biases = 0.01

#     from src.torchfm.torch_utils.options_to_run import Option2Run
#     option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, learning_rate, opt_name, batch_size, reg_coef_vectors, reg_coef_biases, 0)

#     top_main_for_option_run(study, trial, device_ind, option_to_run)

def run_save_single_tensorfm_model_compas():
    opt_name = "adagrad"
    batch_size = 10
    learning_rate = float(0.09789094044298524)
    model_name = tensorfm
    device_ind = 0
    metric_to_optimize = "logloss"
    rank_param = 0
    emb_size = 8
    dim_int = [2, 3]
    ten_ranks = [4, 4]
    study = None
    trial = None
    reg_coef_vectors = 6.787145381076653e-05
    reg_coef_biases = 3.9281412499771244e-05
    option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, learning_rate, opt_name,
                               batch_size, TensorFMParams(dim_int, ten_ranks),
                               reg_coef_vectors, reg_coef_biases, 0.0)
    _, model = top_main_for_option_run(study, trial, device_ind, option_to_run)
    torch.save(model.state_dict(), f"{tmp_save_dir}/tensorfm_model_compas.pt")


def run_save_single_tensorfm_model():
    opt_name = "adagrad"
    batch_size = 1024
    learning_rate = float(0.0963042777)
    model_name = tensorfm
    device_ind = 0
    metric_to_optimize = "logloss"
    rank_param = 0
    emb_size = 8
    dim_int = [2, 3]  # dim_int = [2,3,4]
    ten_ranks = [2, 2]  # ten_ranks = [16,16,16]
    study = None
    trial = None
    reg_coef_vectors = 6.782664216257058e-05
    reg_coef_biases = 4.450358456996342e-05

    option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, learning_rate, opt_name,
                               batch_size, TensorFMParams(dim_int, ten_ranks), reg_coef_vectors, reg_coef_biases, 0.0)
    _, model = top_main_for_option_run(study, trial, device_ind, option_to_run)
    torch.save(model.state_dict(), f"{tmp_save_dir}/tensorfm_model.pt")

    #     Study tensorfm logloss 0 16,tensorfm,logloss,0,16,0.09630427772973353,adagrad,1024,2_3_4_16_16_16,True,6.782664216257058e-05,4.450358456996342e-05,0.0,47,10,0.36920858074128965,0.7984293103218079,0.38045444269295015,0.7793753147125244,0.37957595753737977,0.7798559665679932,6.044466257095337,3.282675266265869,bcelogitloss,avazu

    #     str(self.length) + "," + ",".join([str(i) for i in self.dim_int]) + "," + ",".join([str(i) for i in self.ten_ranks]) # tensorfm
    #     res = ",".join([str(i) for i in [self.m_to_check, self.met_to_opt, self.rank, self.emb_size, self.lr, self.opt_name, self.batch_size, tensor_fm_csv, self.return_l2, self.reg_coef_vectors, self.reg_coef_biases, self.part_id]])
    #     save_all_args_to_file(option_to_run, study_name, option_to_run.to_csv_res(), trial_number, epoch_i, train_err,
    #                               train_auc, valid_err, valid_auc, test_err, test_auc, train_time, valid_time,
    #                               criterion_name, dataset_nm)
    #     study_name, m_to_check, met_to_opt, rank, emb_size, lr, self.opt_name, self.batch_size, tensor_fm_csv, self.return_l2, self.reg_coef_vectors, self.reg_coef_biases, self.part_id

    option_to_run = Option2Run(model_name, metric_to_optimize, rank_param, emb_size, learning_rate, opt_name,
                               batch_size, TensorFMParams(dim_int, ten_ranks), reg_coef_vectors, reg_coef_biases, 0.0)
    _, model = top_main_for_option_run(study, trial, device_ind, option_to_run)
    torch.save(model.state_dict(), f"{tmp_save_dir}/tensorfm_model.pt")


def show_image(arr):
    # transform = T.ToPILImage()
    # img = transform(field_inter_weights)
    # img.show()

    import matplotlib.pyplot as plt
    import numpy as np

    # arr = np.ndarray((1,80,80,1))#This is your tensor
    # fig, ax = plt.subplots(figsize = (40,40))
    # ax.set_title('Interaction Weights of FwFM')
    # ax.set_xlabel('Rows')
    # ax.set_ylabel('Columns')

    arr_ = np.squeeze(arr)  # you can give axis attribute if you wanna squeeze in specific dimension
    plt.imshow(arr_)

    # Add colorbar
    min = np.min(arr_)
    max = np.max(arr_)
    middle = (max + min) / 2
    cbar = plt.colorbar(ticks=[min, middle, max])
    # cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])

    plt.title("Interaction Weights of FwFM")
    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.show()


# field_inter_weights = load_tensor("fwfm_0157_8")
# arr = field_inter_weights.detach().numpy()
# show_image(arr)

if __name__ == '__main__':
    run_save_single_tensorfm_model()
    # run_save_single_tensorfm_model_compas()
