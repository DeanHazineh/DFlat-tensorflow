import tensorflow as tf
import numpy as np
import os
import pickle
from keras_flops import get_flops
import matplotlib.pyplot as plt

import tools.graphFunc as graphFunc
from .neural_utilities import get_flops_alternate
from datasets_metasurface_cells import libraryClass as library
from .neural_utilities import leakyrelu100

mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600",
    "MLP_Nanocylinders_Dense128_U180_H600",
    "MLP_Nanocylinders_Dense64_U180_H600",
    "MLP_Nanofins_Dense1024_U350_H600",
    "MLP_Nanofins_Dense512_U350_H600",
    "MLP_Nanofins_Dense256_U350_H600",
    "MLP_Nanofins_Dense128_U350_H600",
    "MLP_Nanofins_Dense_64_U350_H600",
    "MLP_Nanofins_Dense_32_U350_H600",
]


## BASE CLASS FOR LIBRARY MODELS (PARENT - DO NOT ALTER THIS UNLESS YOU KNOW THE DETAILS)
class MLP_Object(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLP_Object, self).__init__(**kwargs)

        # Define class variables
        self._modelSavePath = ""
        self.trainingLoss = []
        self.trainingValLoss = []
        self.__model_name = ""
        self.__accepts_wavelength = True
        self._dtype = tf.float64
        self.__input_shape_tuple = 0
        self.__output_pol_state = 1

        # parameter limits wrapped into a list for generalized model usage
        self.__preprocessDataBounds = []
        self.__dataBoundsLabel = []
        self._arch = []

    def set_modelSavePath(self, modelSavePath):
        self._modelSavePath = modelSavePath

        if not os.path.exists(self._modelSavePath):
            os.makedirs(modelSavePath)
            os.makedirs(modelSavePath + "/trainingOutput/")

        # Make folders for images too
        if not os.path.exists(self._modelSavePath + "trainingOutput/png_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/png_images/")

        if not os.path.exists(self._modelSavePath + "trainingOutput/pdf_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/pdf_images/")

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

    def set_input_shape(self, input_shape):
        self.__input_shape_tuple = input_shape
        return

    def get_input_shape(self):
        return self.__input_shape_tuple

    def set_output_pol_state(self, output_stack_num):
        self.__output_pol_state = output_stack_num
        return

    def get_output_pol_state(self):
        return self.__output_pol_state

    def call(self, y):
        for layer in self._arch:
            y = layer(y)

        y = tf.cast(y, dtype=self._dtype)
        return y

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
        tf.debugging.assert_equal(
            self.__dataBoundsLabel[-1],
            "wavelength_m",
            message="wavelength should have been the last listed parameter",
            name="preprocessDataBound format assertion",
        )

        wavelength_preprocessBounds = self.__preprocessDataBounds[-1]
        wavelength_mlp = (wavelength_m - wavelength_preprocessBounds[0]) / (
            wavelength_preprocessBounds[1] - wavelength_preprocessBounds[0]
        )

        return wavelength_mlp

    def convert_vectorParam_toMLPInput(self, paramList_asvector):
        ### Sometimes desire mlp output in meshgrid form with vector axis labels.
        # this is just a convenient wrapper to call mlp output on a grid without having
        # to call meshgrid in the main script.
        paramlist_asgrid = np.meshgrid(*paramList_asvector)
        outParams = self.normalizeInput(paramlist_asgrid)

        return np.stack([param.flatten() for param in outParams], -1)

    def customLoadCheckpoint(self):
        ## Custom models require their own functions to handle loads and saves

        # If a checkpoint file exists then load the checkpoint weights to architecture
        print("Checking for model checkpoint at: " + self._modelSavePath)
        if os.path.exists(self._modelSavePath + "checkpoint"):
            self.load_weights(self._modelSavePath).expect_partial()
            print("\n Model Checkpoint Loaded \n")

        # Load the previous training loss vector if it exists
        if os.path.exists(self._modelSavePath + "trainingHistory.pickle"):
            with open(self._modelSavePath + "trainingHistory.pickle", "rb") as handle:
                trackHistory = pickle.load(handle)
                self.trainingLoss = trackHistory["trainingLoss"]
                self.trainingValLoss = trackHistory["trainingValLoss"]

        return

    def customSaveCheckpoint(self, trackHistoryObject=[]):
        # save weights to checkpoint file
        self.save_weights(self._modelSavePath)
        print("\n Model Saved \n")

        # if trackHistory keras object is passed then manually update by concatenating
        # loss vector to current model loss vector and saving current state to pickle
        # Also save a plot displaying the loss state during training for convenience!
        if trackHistoryObject:
            self.trainingLoss = np.concatenate((self.trainingLoss, trackHistoryObject.history["loss"]))
            self.trainingValLoss = np.concatenate((self.trainingValLoss, trackHistoryObject.history["val_loss"]))

            data = {
                "trainingLoss": self.trainingLoss,
                "trainingValLoss": self.trainingValLoss,
            }
            pickle.dump(data, open(self._modelSavePath + "trainingHistory.pickle", "wb"))

            fig = plt.figure(figsize=(20, 10))
            ax = graphFunc.addAxis(fig, 1, 2)
            ax[0].plot(self.trainingLoss, "b-.", label="training loss")
            ax[0].plot(self.trainingValLoss, "r-.", label="validation loss")
            graphFunc.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addlegend=True)

            ax[1].plot(np.log10(self.trainingLoss), "b-.")
            ax[1].plot(np.log10(self.trainingValLoss), "r-.")
            graphFunc.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")

            plt.savefig(self._modelSavePath + "/trainingOutput/png_images/trainingLog_traininghistory.png")
            plt.savefig(self._modelSavePath + "/trainingOutput/pdf_images/trainingLog_traininghistory.pdf")
            plt.close()

        return

    def profile_FLOPs(self):
        # To use keras-flops, we need the architecture defined via keras sequential
        layers = []
        layers.append(tf.keras.Input(shape=self.__input_shape_tuple))

        for layer in self._arch:
            layers.append(layer)

        # Use keras Flops as one metric
        tempModel = tf.keras.Sequential(layers)
        estFLOPs = get_flops(tempModel, batch_size=1)
        print("FLOPs Analysis 1: Keras_Flops: ", estFLOPs)
        print("\n ======================================= \n ")

        # Use second function
        print("The v2 FLOPs is:{}".format(get_flops_alternate(tempModel)), flush=True)

        return estFLOPs


## SUB-BASE: (CHILD - DO NOT ALTER THIS UNLESS YOU KNOW THE DETAILS; ADD NEW CHILDREN FOR DIFFERENT METALIBRARIES)
class MLP_Nanofins_U350_H600(MLP_Object):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_U350_H600, self).__init__(**kwargs)

        # Define model input normalization during training/inference
        # Units in m; These are private class variables and should not be altered unless
        # the corresponding library class was altered
        # NOTE: this is hardcoded here rather than loading directly from library because we
        # do not want the computational/memory cost of loading the library when model is
        # used for inference only!
        __param1Limits = [60e-9, 300e-9]  # corresponds to length x m for data
        __param2Limits = [60e-9, 300e-9]  # corresponds to length y m for data
        __param3Limits = [310e-9, 750e-9]  # corresponds to wavelength m
        paramLimit_labels = ["lenx_m", "leny_m", "wavelength_m"]
        self.set_preprocessDataBounds([__param1Limits, __param2Limits, __param3Limits], paramLimit_labels)

        self.set_input_shape((3,))
        self.set_output_pol_state(2)
        # Define an example architecture (Keras 2.0)
        self._arch = [
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanofins_U350nm_H600nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmission = useLibrary.transmission

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[0, :, :, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[0, :, :, :]).flatten(),  # sin of phase x polarized light
                np.cos(phase[1, :, :, :]).flatten(),  # cos of phase y polarized light
                np.sin(phase[1, :, :, :]).flatten(),  # sin of phase y polarized light
                transmission[0, :, :, :].flatten(),  # x transmission
                transmission[1, :, :, :].flatten(),  # y transmission
            ],
            -1,
        )

        return trainx, trainy

    def get_trainingParam(self):
        useLibrary = library.Nanofins_U350nm_H600nm()
        return useLibrary.params

    def convert_output_complex(self, y_model, reshapeToSize=None):
        phasex = tf.math.atan2(y_model[:, 1], y_model[:, 0])
        phasey = tf.math.atan2(y_model[:, 3], y_model[:, 2])
        transx = y_model[:, 4]
        transy = y_model[:, 5]

        # allow an option to reshape to a grid size (excluding data stack width)
        if reshapeToSize is not None:
            phasex = tf.reshape(phasex, reshapeToSize)
            transx = tf.reshape(transx, reshapeToSize)
            phasey = tf.reshape(phasey, reshapeToSize)
            transy = tf.reshape(transy, reshapeToSize)

            return tf.squeeze(tf.stack([transx, transy]), 1), tf.squeeze(tf.stack([phasex, phasey]), 1)

        return tf.stack([transx, transy]), tf.stack([phasex, phasey])


class MLP_Nanocylinders_U180_H600(MLP_Object):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_U180_H600, self).__init__(**kwargs)

        # Define model input normalization during training/inference
        # Units in m; These are private class variables and should not be altered unless
        # the corresponding library class was altered
        # NOTE: this is hardcoded here rather than loading directly from library because we
        # do not want the computational/memory cost of loading the library when model is
        # used for inference only!
        __param1Limits = [30e-9, 150e-9]  # corresponds to radius m of cylinder for data
        __param2Limits = [310e-9, 750e-9]  # corresponds to wavelength m for training data
        paramLimit_labels = ["radius_m", "wavelength_m"]
        self.set_preprocessDataBounds([__param1Limits, __param2Limits], paramLimit_labels)

        self.set_input_shape((2,))
        self.set_output_pol_state(1)
        # # Define an example architecture (Keras 2.0)
        # Input: wavelength, radius of pillar
        # Output: cos(phase), sin(phase), transmission
        self._arch = [
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanocylinders_U180nm_H600nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmission = useLibrary.transmission

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[:, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[:, :]).flatten(),  # sin of phase x polarized light
                transmission[:, :].flatten(),  # x transmission
            ],
            -1,
        )

        return trainx, trainy

    def get_trainingParam(self):
        useLibrary = library.Nanocylinders_U180nm_H600nm()
        return useLibrary.params

    def convert_output_complex(self, y_model, reshapeToSize=None):
        phasex = tf.math.atan2(y_model[:, 1], y_model[:, 0])
        transx = y_model[:, 2]

        # allow an option to reshape to a grid size (excluding data stack width)
        if reshapeToSize is not None:
            phasex = tf.reshape(phasex, reshapeToSize)
            transx = tf.reshape(transx, reshapeToSize)

        return transx, phasex


## USABLE MLP MODELS
class MLP_Nanocylinders_Dense256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense256_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense128_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense128_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense128_U180_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense128_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                128, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                128, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense64_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense64_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense64_U180_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense64_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense1024_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                1024, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                1024, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense512_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense512_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense512_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense512_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                512, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                512, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense256_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense256_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense256_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense256_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense128_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense128_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense128_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense128_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                128, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                128, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense_64_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense_64_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense_64_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense_64_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense_32_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense_32_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense_32_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense_32_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                32, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                32, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

