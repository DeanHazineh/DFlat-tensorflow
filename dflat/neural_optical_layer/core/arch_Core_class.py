import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import dflat.plot_utilities.graphFunc as graphFunc

### DO NOT ALTER ANYTHING IN THIS FILE IF YOU DON"T KNOW WHAT YOU ARE DOING ELSE IT WILL BREAK THINGS


def get_current_path(folder_name: str):
    resource_path = Path(__file__).parent
    return str(resource_path.joinpath(folder_name)) + "/"


class MLP_Object(tf.keras.Model):
    def __init__(self):
        super(MLP_Object, self).__init__()

        # Define class variables
        self._modelSavePath = ""
        self._dtype = tf.float64  # Output dtype while the model itself is kept on the float32 standard
        self.trainingLoss = []
        self.testLoss = []
        self.__model_name = ""
        self.__accepts_wavelength = True
        self.__input_dim = 1
        self.__output_dim = 1
        self.__output_pol_state = 1

        # parameter limits wrapped into a list for generalized model usage
        self.__preprocessDataBounds = []
        self.__dataBoundsLabel = []
        self._arch = []

    def call(self, y):
        for layer in self._arch:
            y = layer(y)

        if self._dtype != tf.float32:
            y = tf.cast(y, dtype=self._dtype)
        return y

    ###
    def set_arch(self, model_list):
        self._arch = model_list
        return

    def set_model_dtype(self, dtype):
        self._dtype = dtype
        return

    def get_model_dtype(self):
        return self._dtype

    def set_modelSavePath(self, modelSavePath):
        modelSavePath = get_current_path(modelSavePath)
        self._modelSavePath = modelSavePath

        if not os.path.exists(modelSavePath):
            os.makedirs(modelSavePath, exist_ok=True)
            os.makedirs(modelSavePath + "/trainingOutput/", exist_ok=True)

        if not os.path.exists(modelSavePath + "trainingOutput/png_images/"):
            os.makedirs(modelSavePath + "trainingOutput/png_images/", exist_ok=True)

        if not os.path.exists(modelSavePath + "trainingOutput/pdf_images/"):
            os.makedirs(modelSavePath + "trainingOutput/pdf_images/", exist_ok=True)

        return

    def set_preprocessDataBounds(self, preprocessDataBounds, boundLabels):
        self.__preprocessDataBounds = preprocessDataBounds
        self.__dataBoundsLabel = boundLabels
        return

    def get_preprocessDataBounds(self):
        return self.__preprocessDataBounds

    def set_model_name(self, name):
        self.__model_name = name
        return

    def get_model_name(self):
        return self.__model_name

    def set_wavelengthFlag(self, boolFlag):
        self.__accepts_wavelength = boolFlag
        return

    def get_wavelengthFlag(self):
        return self.__accepts_wavelength

    def set_input_shape(self, input_dim):
        self.__input_dim = input_dim
        return

    def get_input_shape(self):
        return self.__input_dim

    def set_output_shape(self, output_dim):
        self.__output_dim = output_dim
        return

    def get_output_shape(self):
        return self.__output_dim

    def set_output_pol_state(self, output_stack_num):
        self.__output_pol_state = output_stack_num
        return

    def get_output_pol_state(self):
        return self.__output_pol_state

    def normalizeInput(self, paramList):
        # take in a list of parameters and normalize
        # based on the class pre-defined parameter limits
        # Normalized input parameters to [0,1]
        # Ensures stability with NN initialization and compatability with constrained optimization
        outParams = []

        for counter, thisParam in enumerate(paramList):
            parameterBounds = self.__preprocessDataBounds[counter]
            outParams.append((thisParam - parameterBounds[0]) / (parameterBounds[1] - parameterBounds[0]))

        return outParams

    def normalizeWavelength(self, wavelength_m):
        wavelength_preprocessBounds = self.__preprocessDataBounds[-1]
        wavelength_mlp = (wavelength_m - wavelength_preprocessBounds[0]) / (wavelength_preprocessBounds[1] - wavelength_preprocessBounds[0])

        return wavelength_mlp

    def convert_vectorParam_toMLPInput(self, paramList_asvector):
        ### Sometimes desire mlp output in meshgrid form with vector axis labels.
        # this is just a convenient wrapper to call mlp output on a grid without having
        # to call meshgrid in the main script.
        paramlist_asgrid = np.meshgrid(*paramList_asvector)
        outParams = self.normalizeInput(paramlist_asgrid)

        return np.stack([param.flatten() for param in outParams], -1)

    def customSaveCheckpoint(self, test_loss=[], training_loss=[], verbose=False):
        # save weights to checkpoint file
        self.save_weights(self._modelSavePath)
        if verbose:
            print(f"Model checkpoint saved to: ", self._modelSavePath)

        # if Losses are passed then manually update by concatenating
        if training_loss:
            self.trainingLoss = np.concatenate((self.trainingLoss, training_loss))
        if test_loss:
            self.testLoss = np.concatenate((self.testLoss, test_loss))

        data = {
            "loss": self.trainingLoss,
            "test_loss": self.testLoss,
        }
        pickle.dump(data, open(self._modelSavePath + "trainingHistory.pickle", "wb"))

        fig = plt.figure(figsize=(10, 5))
        ax = graphFunc.addAxis(fig, 1, 2)
        ax[0].plot(self.trainingLoss, "b-.", label="training loss")
        ax[0].plot(self.testLoss, "r-.", label="test loss")
        graphFunc.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addLegend=True)

        ax[1].plot(np.log10(self.trainingLoss), "b-.")
        ax[1].plot(np.log10(self.testLoss), "r-.")
        graphFunc.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")

        plt.savefig(self._modelSavePath + "/trainingOutput/png_images/trainingLog_traininghistory.png")
        plt.savefig(self._modelSavePath + "/trainingOutput/pdf_images/trainingLog_traininghistory.pdf")
        plt.close()
        return

    def customLoadCheckpoint(self):
        ## Custom models require their own functions to handle loads and saves
        # If a checkpoint file exists then load the checkpoint weights to architecture
        print("Checking for model checkpoint at: " + self._modelSavePath)
        if os.path.exists(self._modelSavePath + "checkpoint"):
            self.load_weights(self._modelSavePath).expect_partial()
            print("\n Model Checkpoint Loaded \n")
        else:
            print("\n no model checkpoint found at\n", self._modelSavePath + "checkpoint")

        # Load the previous training loss vector if it exists
        if os.path.exists(self._modelSavePath + "trainingHistory.pickle"):
            with open(self._modelSavePath + "trainingHistory.pickle", "rb") as handle:
                trackHistory = pickle.load(handle)
                self.trainingLoss = trackHistory["loss"]
                self.testLoss = trackHistory["test_loss"]

        return
