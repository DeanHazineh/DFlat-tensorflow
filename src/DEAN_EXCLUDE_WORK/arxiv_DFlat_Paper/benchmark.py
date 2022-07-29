import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(".")
import tools.graphFunc as gF


def median_stats(sorted_data):
    num_pts = len(sorted_data)

    fquart = sorted_data[int(num_pts * 0.25)]
    fbench3 = sorted_data[int(num_pts * 0.50)]
    thirdquart = sorted_data[int(num_pts * 0.75)]

    return [fquart, fbench3, thirdquart]


def load_model_get_stat(path_to_model, dat_string, stat_fun):

    dat_path = path_to_model + "/training_testDataError.pickle"
    file = open(dat_path, "rb")
    data = pickle.load(file)
    file.close()

    # account for polarization vs nonpolarization arrays
    thisdat = data[dat_string]
    if len(thisdat.shape) == 1:
        thisdat = np.expand_dims(thisdat, 0)

    abs_error = np.mean(np.abs(thisdat), axis=0)
    absError_sort = np.sort(abs_error)

    dat_stat = stat_fun(absError_sort)
    log_flops = np.log10(data["est_FLOPs"])

    return dat_stat, log_flops


def sweep_files_stats(fold_path, file_name_list, stat_fun, dat_string="complex_error"):

    stat_array = []
    log_flop_array = []
    for file_name in file_name_list:
        stat, log_flops = load_model_get_stat(fold_path + file_name, dat_string, stat_fun=stat_fun)
        stat_array.append(stat)
        log_flop_array.append(log_flops)

    return np.stack(stat_array), np.stack(log_flop_array)


def generate_nanofin_benchmark_figure():
    ### Define Data Paths
    trained_mlp_path = "neural_optical_layer/core/trained_MLP_models/"
    mlp_model_names = [
        "MLP_Nanofins_Dense32_U350_H600",
        "MLP_Nanofins_Dense64_U350_H600",
        "MLP_Nanofins_Dense128_U350_H600",
        "MLP_Nanofins_Dense256_U350_H600",
        "MLP_Nanofins_Dense512_U350_H600",
        "MLP_Nanofins_Dense1024_U350_H600",
    ]

    trained_erbf_path = "neural_optical_layer/core/trained_erbf_models/"
    erbf_model_names = [
        "ERBF_Nanofins_B128_U350_H600",
        "ERBF_Nanofins_B256_U350_H600",
        "ERBF_Nanofins_B512_U350_H600",
        "ERBF_Nanofins_B1024_U350_H600",
        "ERBF_Nanofins_B2048_U350_H600",
    ]

    regress_multipoly_path = "neural_optical_layer/core/baseline_regression/fitted_regression_models/"
    multipoly_model_names = [
        "multipoly_nanofins_6",
        "multipoly_nanofins_7",
        "multipoly_nanofins_8",
        "multipoly_nanofins_9",
        "multipoly_nanofins_10",
        "multipoly_nanofins_11",
        "multipoly_nanofins_12",
        "multipoly_nanofins_13",
        "multipoly_nanofins_14",
        "multipoly_nanofins_15",
        "multipoly_nanofins_16",
    ]

    ### Grab Model Statistics
    mlp_stats, mlp_log_flops = sweep_files_stats(
        trained_mlp_path, mlp_model_names, median_stats, dat_string="complex_error"
    )
    ebf_stats, ebf_log_flops = sweep_files_stats(
        trained_erbf_path, erbf_model_names, median_stats, dat_string="complex_error"
    )
    poly_stats, poly_log_flops = sweep_files_stats(
        regress_multipoly_path, multipoly_model_names, median_stats, dat_string="complex_error"
    )

    ### Make Figure
    dataXPlot = [mlp_log_flops, ebf_log_flops, poly_log_flops]
    dataYPlot = [mlp_stats, ebf_stats, poly_stats]
    color = ["blue", "green", "red"]

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 1)
    for i in range(len(dataXPlot)):
        xdat = dataXPlot[i]
        ydat = dataYPlot[i]
        useColor = color[i]
        for j in range(ydat.shape[1]):
            ax[0].plot(xdat, ydat[:, j], "x-", c=useColor, alpha=1.0)

        ax[0].fill_between(xdat, ydat[:, 1], ydat[:, 0], color=useColor, alpha=0.5)
        ax[0].fill_between(xdat, ydat[:, 2], ydat[:, 1], color=useColor, alpha=0.5)

    plt.show()

    return


def generate_nanocylinder_benchmark_figure():
    ### Define Data Paths
    trained_mlp_path = "neural_optical_layer/core/trained_MLP_models/"
    mlp_model_names = [
        "MLP_Nanocylinders_Dense32_U180_H600",
        "MLP_Nanocylinders_Dense64_U180_H600",
        "MLP_Nanocylinders_Dense128_U180_H600",
        "MLP_Nanocylinders_Dense256_U180_H600",
    ]

    trained_erbf_path = "neural_optical_layer/core/trained_erbf_models/"
    erbf_model_names = [
        "ERBF_Nanocylinders_B32_U180_H600",
        "ERBF_Nanocylinders_B64_U180_H600",
        "ERBF_Nanocylinders_B128_U180_H600",
        "ERBF_Nanocylinders_B256_U180_H600",
        "ERBF_Nanocylinders_B512_U180_H600",
    ]

    regress_multipoly_path = "neural_optical_layer/core/baseline_regression/fitted_regression_models/"
    multipoly_model_names = [
        "multipoly_nanocylinders_10",
        "multipoly_nanocylinders_14",
        "multipoly_nanocylinders_18",
        "multipoly_nanocylinders_22",
        "multipoly_nanocylinders_26",
        "multipoly_nanocylinders_30",
        "multipoly_nanocylinders_34",
    ]

    ### Grab Model Statistics
    mlp_stats, mlp_log_flops = sweep_files_stats(
        trained_mlp_path, mlp_model_names, median_stats, dat_string="complex_error"
    )
    ebf_stats, ebf_log_flops = sweep_files_stats(
        trained_erbf_path, erbf_model_names, median_stats, dat_string="complex_error"
    )
    poly_stats, poly_log_flops = sweep_files_stats(
        regress_multipoly_path, multipoly_model_names, median_stats, dat_string="complex_error"
    )

    ### Make Figure
    dataXPlot = [mlp_log_flops, ebf_log_flops, poly_log_flops]
    dataYPlot = [mlp_stats, ebf_stats, poly_stats]
    color = [
        "blue",
        "green",
        "red",
    ]

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 1)
    for i in range(len(dataXPlot)):
        xdat = dataXPlot[i]
        ydat = dataYPlot[i]
        useColor = color[i]
        for j in range(ydat.shape[1]):
            ax[0].plot(xdat, ydat[:, j], "x-", c=useColor, alpha=1.0)

        ax[0].fill_between(xdat, ydat[:, 1], ydat[:, 0], color=useColor, alpha=0.5)
        ax[0].fill_between(xdat, ydat[:, 2], ydat[:, 1], color=useColor, alpha=0.5)

    plt.show()

    return


if __name__ == "__main__":
    generate_nanofin_benchmark_figure()
    generate_nanocylinder_benchmark_figure()
