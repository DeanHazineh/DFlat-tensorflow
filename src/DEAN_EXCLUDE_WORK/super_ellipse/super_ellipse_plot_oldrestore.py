import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(".")
import tools.graphFunc as gF

if __name__ == "__main__":
    ## Get file names
    # datapath = "cell_library_generation/dev_test/output/super_ellipse/"
    # file_names = []
    # for filename in os.listdir(datapath):
    #     if "11b.pickle" in filename:
    #         file_names.append(filename)

    # # load data
    # trans = []
    # phase = []
    # paramlist = []
    # wavelength = []
    # n_val = [1.0, 2.0, 3.0, 4.0, 10.0]
    # for filename in file_names:
    #     print(filename)

    #     file = open(datapath + filename, "rb")
    #     data = pickle.load(file)

    #     trans.append(data["trans"])
    #     phase.append(data["phase"])
    #     paramlist.append(data["paramlist"])
    #     wavelength.append(data["wavelength_set_m"])

    # load data
    datapath = "cell_library_generation/dev_test/output/super_ellipse/"

    trans = []
    phase = []
    paramlist = []
    wavelength = []
    # n_vec = [1.0, 2.0, 3.0, 4.0, 10.0]
    n_vec = [2.0, 3.0, 4.0, 10.0]

    for n_val in n_vec:
        file = open(datapath + "run_superEllipse_Sweep1D_n" + str(n_val) + "_FM11b.pickle", "rb")
        data = pickle.load(file)

        trans.append(data["trans"])
        phase.append(data["phase"])
        paramlist.append(data["paramlist"])
        wavelength.append(data["wavelength_set_m"])

    trans = np.vstack(trans)
    phase = np.vstack(phase)
    len_vec = paramlist[0][:, 0] * 1e9

    # Make plot
    norm = mpl.colors.Normalize(vmin=min(n_vec), vmax=max(n_vec))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 1)
    wl_idx = 2
    for n_idx in range(trans.shape[0]):
        rgba = np.expand_dims(np.array(cmap.to_rgba(n_vec[n_idx])), 0)
        ax[0].scatter(len_vec, trans[n_idx, :, wl_idx, 0], s=10, marker="o", lw=3, c=rgba)
        ax[0].plot(len_vec, trans[n_idx, :, wl_idx, 0], "x-", c=cmap.to_rgba(n_vec[n_idx]), lw=3, alpha=0.5)
    fig.colorbar(cmap, ticks=n_vec)

    fig = plt.figure()
    ax = gF.addAxis(fig, 1, 1)
    wl_idx = 2
    for n_idx in range(trans.shape[0]):
        rgba = np.expand_dims(np.array(cmap.to_rgba(n_vec[n_idx])), 0)
        ax[0].scatter(len_vec, phase[n_idx, :, wl_idx, 0], s=10, marker="o", lw=3, c=rgba)
        ax[0].plot(len_vec, phase[n_idx, :, wl_idx, 0], "x-", c=cmap.to_rgba(n_vec[n_idx]), lw=3, alpha=0.5)
    fig.colorbar(cmap, ticks=n_vec)

    plt.show()

    # fig = plt.figure()
    # ax = gF.addAxis(fig, 1, 5)
    # for wl_idx in range(5):
    #     for n_idx in range(trans.shape[0]):
    #         ax[wl_idx].plot(trans[n_idx, :, wl_idx, 0], "x-", c=cmap.to_rgba(n_vec[n_idx]), lw=3)

    # fig.colorbar(cmap, ticks=n_vec)
    # plt.show()
