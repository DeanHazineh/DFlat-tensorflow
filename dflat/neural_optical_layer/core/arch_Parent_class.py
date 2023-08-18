import tensorflow as tf
import numpy as np

from dflat.metasurface_library import libraryClass as library
from .arch_Core_class import MLP_Object


## SUB-BASE: (CHILD - DO NOT ALTER THIS UNLESS YOU KNOW THE DETAILS; ADD NEW CHILDREN FOR DIFFERENT METALIBRARIES)
class MLP_Nanofins_U350_H600(MLP_Object):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_U350_H600, self).__init__()

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
        self.set_model_dtype(dtype)
        self.set_input_shape(3)
        self.set_output_shape(6)
        self.set_output_pol_state(2)
        return

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanofins_U350nm_H600nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmittance = useLibrary.transmittance

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[0, :, :, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[0, :, :, :]).flatten(),  # sin of phase x polarized light
                np.cos(phase[1, :, :, :]).flatten(),  # cos of phase y polarized light
                np.sin(phase[1, :, :, :]).flatten(),  # sin of phase y polarized light
                transmittance[0, :, :, :].flatten(),  # x transmission
                transmittance[1, :, :, :].flatten(),  # y transmission
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
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_U180_H600, self).__init__()

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
        self.set_model_dtype(dtype)
        self.set_input_shape(2)
        self.set_output_shape(3)
        self.set_output_pol_state(1)
        return

    def returnLibraryAsTrainingData(self):
        # FDTD generated data loaded from library class file
        useLibrary = library.Nanocylinders_U180nm_H600nm()
        params = useLibrary.params
        phase = useLibrary.phase
        transmittance = useLibrary.transmittance

        # Normalize inputs (always done based on self model normalize function)
        normalizedParams = self.normalizeInput(params)
        trainx = np.stack([param.flatten() for param in normalizedParams], -1)
        trainy = np.stack(
            [
                np.cos(phase[:, :]).flatten(),  # cos of phase x polarized light
                np.sin(phase[:, :]).flatten(),  # sin of phase x polarized light
                transmittance[:, :].flatten(),  # x transmission
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
