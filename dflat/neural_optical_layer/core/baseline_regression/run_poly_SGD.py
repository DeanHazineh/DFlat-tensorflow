# -*- coding: utf-8 -*-
# @Author: Dean Hazineh, Harvard University
# @Date:   2022-06-01 12:49:29
# @Last Modified by:   Dean Hazineh, Harvard University
# @Last Modified time: 2022-06-01 21:43:26

import sys

sys.path.append(".")

from sklearn.model_selection import train_test_split
from interpolant_optical_layer_INDEVELOPMENT.core.multivariate_polynomial_SGD import *
import pickle
import dflat.plot_utilities.graphFunc as gF

# Here, we piggy back off the base class training data functions to train other proxy models
# NOTE: In the future, it may make sense to remove the return training data function to outside the model class so that
# we don't need to call the model class here next time.
from neural_optical_layer.core.mlp_models import MLP_Nanocylinders_U180_H600, MLP_Nanofins_U350_H600


def grid_search_polynomial_degree(model_name, modelClass, folder):
    polyDegree = np.arange(2, 40, 5)

    # Loop over polyDegree single regression fits
    train_loss = []
    val_loss = []
    for degree in polyDegree:
        print("Sweep Poly Degree: ", degree)
        with tf.device("/gpu:0"):
            track_history = single_poly_SGD_train(model_name, modelClass, degree, term_grad_loss=1.0e-5, use_ckpt=False)

        train_loss.append(track_history["loss"])
        val_loss.append(track_history["val_loss"])
        checkpoint_save = {"polyDegree": polyDegree, "loss": train_loss, "val_loss": val_loss}
        with open(folder + "checkpoint_sweepDat.pickle", "wb") as handle:
            pickle.dump(checkpoint_save, handle)

    # Save the sweep data outcomse
    finalLossDat = []
    for iter, degree in enumerate(polyDegree):
        finalLossDat.append([polyDegree[iter], train_loss[iter][-1], val_loss[iter][-1]])

    sweepDat = {"degreeSweepDat": np.vstack(finalLossDat), "train_loss": train_loss, "val_loss": val_loss}
    with open(folder + "sweepDat.pickle", "wb") as handle:
        pickle.dump(sweepDat, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def run_nanofin_train():
    ## Run grid Search
    # folder = "interpolant_optical_layer/core/polynomial_models/sweep_polyDegree_Fins/"
    # grid_search_polynomial_degree(MLP_Nanofins_U350_H600, folder)
    # plot_degree_sweep(folder)

    # # Train single poly for Nanofins
    # single_poly_SGD_train(
    #     "NanoFin_poly",
    #     MLP_Nanocylinders_U180_H600,
    #     polyDegree=15,
    #     lr=1,
    #     term_grad_loss=1e-5,
    #     use_ckpt=False,
    #     max_epochs=500,
    #     verbose=True,
    # )
    # plot_nanofin_polyModel("multi_poly_Nanofins_Degree30")

    return


def run_nanocylinder_train():
    ### Run grid Search
    # folder = "interpolant_optical_layer/dev_testing/sweep_polyDegree_NanoCylinder/"
    # grid_search_polynomial_degree("sweep_nanocylinder", MLP_Nanocylinders_U180_H600, folder)
    # plot_degree_sweep(folder)

    # single_model_train(
    #     "Nanocylinder_poly",
    #     MLP_Nanocylinders_U180_H600,
    #     polyDegree=10,
    #     lr=1,
    #     term_grad_loss=1e-5,
    #     use_ckpt=False,
    #     max_epochs=500,
    #     verbose=True,
    # )
    # poly_regression = multi_poly_regression("Nanocylinder_poly", 10, 2, 3)
    # poly_regression.customLoadCheckpoint()
    # plot_nanocylinder_polyModel(poly_regression)

    return


def plot_degree_sweep(folder):
    # Load the data
    with open(folder + "sweepDat.pickle", "rb") as handle:
        dat = pickle.load(handle)
        degreeSweepDat = dat["degreeSweepDat"]
        train_loss = dat["train_loss"]
        val_loss = dat["val_loss"]

    fig = plt.figure(figsize=(20, 10))
    ax = gF.addAxis(fig, 1, 2)
    ax[0].plot(degreeSweepDat[:, 0], degreeSweepDat[:, 1], "kx--")
    ax[0].plot(degreeSweepDat[:, 0], degreeSweepDat[:, 2], "ro--")

    ax[1].plot(train_loss[-1][:], "k--")
    ax[1].plot(val_loss[-1][:], "r--")
    plt.show()

    return


def plot_nanocylinder_polyModel(regression_model):
    # View the upsampled data
    # input parameters converted back to vector form from meshgrid for convenience
    model = MLP_Nanocylinders_U180_H600()
    trainingParam = model.get_trainingParam()
    param1_r = trainingParam[0][0, :]
    param2_w = trainingParam[1][:, 0]
    nr = len(param1_r)
    nw = len(param2_w)

    ### Create upsampled model prediction on the same bounds
    upsampleFactor = 4
    param1_rpred = np.linspace(np.min(param1_r), np.max(param1_r), upsampleFactor * nr)
    param2_wpred = np.linspace(np.min(param2_w), np.max(param2_w), upsampleFactor * nw)
    mlpinput = model.convert_vectorParam_toMLPInput([param1_rpred, param2_wpred])

    model_pred = regression_model(mlpinput)
    pred_trans, pred_phase = model.convert_output_complex(
        model_pred, reshapeToSize=[1, len(param2_wpred), len(param1_rpred)]
    )
    print(pred_trans.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(121)
    ax.imshow(pred_trans[0], vmin=0, vmax=1)
    ax = fig.add_subplot(122)
    ax.imshow(pred_phase[0])
    plt.show()

    return


def plot_nanofin_polyModel(model_name):
    # Upsample the training data grid and generate a display for model output
    model = MLP_Nanofins_U350_H600()
    trainingParam = model.get_trainingParam()
    poly_regression = multi_poly_regression(model_name, 30, 3, 6)
    poly_regression.customLoadCheckpoint()

    # input parameters converted back to vector form from meshgrid for convenience
    param1_x = trainingParam[0][0, :, 0]
    param2_y = trainingParam[1][:, 0, 0]
    param3_w = trainingParam[2][0, 0, :]
    nx = len(param1_x)
    ny = len(param2_y)

    ### Analyze MLP predictions on upsampled grid in space
    upsampleFactor = 4
    param1_xpred = np.linspace(np.min(param1_x), np.max(param1_x), upsampleFactor * nx)
    param2_ypred = np.linspace(np.min(param2_y), np.max(param2_y), upsampleFactor * ny)

    ### Analyze the Model vs Lumerical for many wavelength slices (upsampled Space)
    w_set = np.array([330e-9, 430e-9, 530e-9, 630e-9, 730e-9])
    w_idx = np.zeros_like(w_set)
    for wi, w in enumerate(w_set):
        w_idx[wi] = np.argmin(np.abs(w - param3_w))

    # Create and save a plot for each wavelength
    for iter in range(len(w_set)):
        wavelength = w_set[iter]
        model_input = model.convert_vectorParam_toMLPInput([param1_xpred, param2_ypred, wavelength])
        model_pred = poly_regression(model_input)

        pred_trans, pred_phase = model.convert_output_complex(
            model_pred, reshapeToSize=[1, len(param2_ypred), len(param1_xpred)]
        )

        fig = plt.figure()
        ax = gF.addAxis(fig, 2, 2)
        ax[0].imshow(pred_trans[0], vmin=0, vmax=1)
        ax[1].imshow(pred_trans[1], vmin=0, vmax=1)
        ax[2].imshow(pred_phase[0])
        ax[3].imshow(pred_phase[1])
        plt.show()

    return


if __name__ == "__main__":
    #    

