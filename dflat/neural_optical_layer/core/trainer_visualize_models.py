import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from dflat.neural_optical_layer.core.models_DNN import *
from dflat.neural_optical_layer.core.models_eRBF import *
import dflat.plot_utilities.graphFunc as graphFunc


def plot_MLP_Nanocylinders(mlp_object):
    ### Load mlp class
    mlp_object.customLoadCheckpoint()

    # Print for reference the computational cost of the model
    mlp_object.profile_FLOPs()

    ### Get the true data used in training and reformat it back to grid
    trainingParam = mlp_object.get_trainingParam()
    _, outputData = mlp_object.returnLibraryAsTrainingData()
    trans_true, phase_true = mlp_object.convert_output_complex(outputData, trainingParam[0].shape)

    # input parameters converted back to vector form from meshgrid for convenience
    param1_r = trainingParam[0][0, :]
    param2_w = trainingParam[1][:, 0]
    nr = len(param1_r)
    nw = len(param2_w)

    ### Create upsampled model prediction on the same bounds
    upsampleFactor = 4
    param1_rpred = np.linspace(np.min(param1_r), np.max(param1_r), upsampleFactor * nr)
    param2_wpred = np.linspace(np.min(param2_w), np.max(param2_w), upsampleFactor * nw)
    mlpinput = mlp_object.convert_vectorParam_toMLPInput([param1_rpred, param2_wpred])
    y_model = mlp_object.predict(mlpinput)
    trans_model, phase_model = mlp_object.convert_output_complex(y_model, reshapeToSize=[len(param2_wpred), len(param1_rpred)])

    ### make plot
    fig = plt.figure(figsize=(30, 30))
    axisList = graphFunc.addAxis(fig, 2, 2)
    im0 = axisList[0].imshow(
        trans_true,
        extent=(min(param1_r * 1e9), max(param1_r * 1e9), max(param2_w * 1e9), min(param2_w * 1e9)),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    im1 = axisList[1].imshow(
        phase_true,
        extent=(min(param1_r * 1e9), max(param1_r * 1e9), max(param2_w * 1e9), min(param2_w * 1e9)),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
    )
    im2 = axisList[2].imshow(
        trans_model,
        extent=(min(param1_rpred * 1e9), max(param1_rpred * 1e9), max(param2_wpred * 1e9), min(param2_wpred * 1e9)),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    im3 = axisList[3].imshow(
        phase_model,
        extent=(min(param1_rpred * 1e9), max(param1_rpred * 1e9), max(param2_wpred * 1e9), min(param2_wpred * 1e9)),
        vmin=-np.pi,
        vmax=np.pi,
        aspect="auto",
    )

    graphFunc.formatPlots(
        fig,
        axisList[0],
        im0,
        "Cylinder radius (nm)",
        "Incident wavelength (nm)",
        "Transmittance (Lumerical)",
        addcolorbar=True,
        cbartitle="Normalized Power",
    )
    graphFunc.formatPlots(
        fig,
        axisList[1],
        im1,
        "Cylinder radius (nm)",
        "Incident wavelength (nm)",
        "Phase shift (Lumerical)",
        addcolorbar=True,
        cbartitle="Phase (radians)",
    )
    graphFunc.formatPlots(
        fig,
        axisList[2],
        im2,
        "Cylinder radius (nm)",
        "Incident wavelength (nm)",
        "Transmittance (MLP)",
        addcolorbar=True,
        cbartitle="Normalized Power",
    )
    graphFunc.formatPlots(
        fig,
        axisList[3],
        im3,
        "Cylinder radius (nm)",
        "Incident wavelength (nm)",
        "Phase shift (MLP)",
        addcolorbar=True,
        cbartitle="Phase (radians)",
    )

    # plt.tight_layout()
    plt.savefig(mlp_object._modelSavePath + "/trainingOutput/png_images/trainedMLP_opticalResponse.png")
    plt.savefig(mlp_object._modelSavePath + "/trainingOutput/pdf_images/trainedMLP_opticalResponse.pdf")
    plt.show()

    return


def plot_MLP_Nanofins(mlp_object):
    # Load saved model
    mlp_object.customLoadCheckpoint()

    # Report FLOPs
    mlp_object.profile_FLOPs()

    ### Get the true data used in training and reformat it back to grid
    trainingParam = mlp_object.get_trainingParam()
    _, outputData = mlp_object.returnLibraryAsTrainingData()
    trans_true, phase_true = mlp_object.convert_output_complex(outputData, np.expand_dims(trainingParam[0], 0).shape)

    # input parameters converted back to vector form from meshgrid for convenience
    param1_x = trainingParam[0][0, :, 0]
    param2_y = trainingParam[1][:, 0, 0]
    param3_w = trainingParam[2][0, 0, :]
    nx, ny = len(param1_x), len(param2_y)

    ### Define upsampled grid to evaluate predictions on
    upsampleFactor = 4
    param1_xpred = np.linspace(np.min(param1_x), np.max(param1_x), upsampleFactor * nx)
    param2_ypred = np.linspace(np.min(param2_y), np.max(param2_y), upsampleFactor * ny)

    ### Compute and compare the MLP vs Lumerical for many wavelength slices (upsampled Space)
    w_set = np.array([330e-9, 430e-9, 530e-9, 630e-9, 730e-9])
    w_idx = np.zeros_like(w_set)
    for wi, w in enumerate(w_set):
        w_idx[wi] = np.argmin(np.abs(w - param3_w))
    w_idx_set = w_idx.astype("int")

    # Create and save a plot for each wavelength
    saveFold = mlp_object._modelSavePath
    extent_orig = (min(param1_x), max(param1_x), max(param2_y), min(param2_y))
    extent_upsample = (min(param1_x), max(param1_x), max(param2_y), min(param2_y))
    for iter, wavelength in enumerate(w_set):

        mlpinput = mlp_object.convert_vectorParam_toMLPInput([param1_xpred, param2_ypred, wavelength])
        y_model = mlp_object.predict(mlpinput)
        trans_mlp, phase_mlp = mlp_object.convert_output_complex(y_model, reshapeToSize=[1, len(param2_ypred), len(param1_xpred), 1])

        # Get corresponding true slice
        trans_trueW = trans_true[:, :, :, w_idx_set[iter]]
        phase_trueW = phase_true[:, :, :, w_idx_set[iter]]

        fig = plt.figure(figsize=(30, 30))
        axisList = graphFunc.addAxis(fig, 2, 2)
        im0 = axisList[0].imshow(trans_trueW[0, :, :], extent=extent_orig, vmin=0, vmax=1)
        im1 = axisList[1].imshow(trans_trueW[1, :, :], extent=extent_upsample, vmin=0, vmax=1)
        im2 = axisList[2].imshow(np.squeeze(trans_mlp[0]), extent=extent_upsample, vmin=0, vmax=1)
        im3 = axisList[3].imshow(np.squeeze(trans_mlp[1]), extent=extent_upsample, vmin=0, vmax=1)
        graphFunc.formatPlots(fig, axisList[0], im0, "Fin length x (nm)", "Fin length y (nm)", "Transmittance x-pol. (Lumerical)")
        graphFunc.formatPlots(
            fig,
            axisList[1],
            im1,
            "Fin length x (nm)",
            "",
            "Transmittance y-pol. (Lumerical)",
            addcolorbar=True,
            rmvyLabel=True,
            cbartitle="Normalized power",
        )
        graphFunc.formatPlots(fig, axisList[2], im2, "Fin len. x (nm)", "Fin len. y (nm)", "Transmittance x-pol. (MLP)")
        graphFunc.formatPlots(
            fig,
            axisList[3],
            im3,
            "Fin lenghth x (nm)",
            "",
            "Transmittance y-pol. (MLP)",
            addcolorbar=True,
            rmvyLabel=True,
            cbartitle="Normalized power",
        )
        filename = "trainedMLP_w" + str((wavelength * 1e9).astype("int")) + "_transmission"
        plt.savefig(saveFold + "/trainingOutput/png_images/" + filename + ".png")
        plt.savefig(saveFold + "/trainingOutput/pdf_images/" + filename + ".pdf")

        fig = plt.figure(figsize=(30, 30))
        axisList = graphFunc.addAxis(fig, 2, 2)
        im0 = axisList[0].imshow(phase_trueW[0, :, :], extent=extent_orig, vmin=-np.pi, vmax=np.pi)
        im1 = axisList[1].imshow(phase_trueW[1, :, :], extent=extent_orig, vmin=-np.pi, vmax=np.pi)
        im2 = axisList[2].imshow(np.squeeze(phase_mlp[0]), extent=extent_upsample, vmin=-np.pi, vmax=np.pi)
        im3 = axisList[3].imshow(np.squeeze(phase_mlp[1]), extent=extent_upsample, vmin=-np.pi, vmax=np.pi)
        graphFunc.formatPlots(fig, axisList[0], im0, "Fin length x (nm)", "Fin length y (nm)", "Phase shift x-pol. (Lumerical)")
        graphFunc.formatPlots(
            fig,
            axisList[1],
            im1,
            "Fin length x (nm)",
            "",
            "Phase shift y-pol. (Lumerical)",
            addcolorbar=True,
            rmvyLabel=True,
            cbartitle="Phase (radians)",
        )
        graphFunc.formatPlots(fig, axisList[2], im2, "Fin length x (nm)", "Fin length y (nm)", "Phase shift x-pol. (MLP)")
        graphFunc.formatPlots(
            fig,
            axisList[3],
            im3,
            "Fin length x (nm)",
            "",
            "Phase shift y-pol. (MLP)",
            addcolorbar=True,
            rmvyLabel=True,
            cbartitle="Phase (radians)",
        )
        filename = "trainedMLP_w" + str((wavelength * 1e9).astype("int")) + "_phase"
        plt.savefig(saveFold + "/trainingOutput/png_images/" + filename + "_phase.png")
        plt.savefig(saveFold + "/trainingOutput/pdf_images/" + filename + "_phase.pdf")

    return


if __name__ == "__main__":
    ## Run plotting functions for mlp model
    # plot_MLP_Nanocylinders(MLP_Nanocylinders_Dense64_U180_H600())
    # plot_MLP_Nanofins(MLP_Nanofins_Dense256_U350_H600())
    # plot_MLP_Nanofins(MLP_Nanofins_Dense512_U350_H600())
    plot_MLP_Nanofins(MLP_Nanofins_GFFDense256_256s1p0_U350_H600())
