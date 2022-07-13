import tensorflow as tf
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras.backend as kend
import tools.graphFunc as gF

# This is required to make pdf figures compatible with adobe illustrator
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


#  BASE CLASS FOR PIPELINES TO USE TRAIN HELPERS
class Pipeline_Object(tf.keras.Model):
    """Baseclass for DFlat custom pipelines, inherits tf.keras.Model structure.

    Attributes:
        `loss_vector` (list): List storing the loss at each epoch after training
        `savepath` (str): Pipeline savepath to store model checkpoints, data, and figures.
        `saveAtEpochs` (int): Number of training epochs between intermediate saves.
    """

    def __init__(self, savepath, saveAtEpochs=None):
        """Initialization for base class

        Args:
            `savepath` (str): Pipeline savepath to store model checkpoints, data, and figures.
            `saveAtEpochs` (int): Number of training epochs between intermediate saves.
        """
        super(Pipeline_Object, self).__init__()

        # Define class variables
        self.loss_vector = []
        self.savepath = savepath
        self.saveAtEpochs = saveAtEpochs

        # create the savepath folder if it does not exist
        self.__checkModelPath()

    def __checkModelPath(self):

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
            os.makedirs(self.savepath + "trainingOutput/")

        # Make folders for images too
        if not os.path.exists(self.savepath + "trainingOutput/png_images/"):
            os.makedirs(self.savepath + "trainingOutput/png_images/")
        if not os.path.exists(self.savepath + "trainingOutput/pdf_images/"):
            os.makedirs(self.savepath + "trainingOutput/pdf_images/")

        return

    def customSaveCheckpoint(self, loss_vector=[]):
        # Save Weights
        self.save_weights(self.savepath)
        print("\n Model Saved Succesfully \n")

        if loss_vector:
            self.loss_vector = np.concatenate((self.loss_vector, loss_vector))
            data = {"trainingLoss": self.loss_vector}
            pickle.dump(data, open(self.savepath + "trainingHistory.pickle", "wb"))

        # Make and save a plot of the training history
        fig = plt.figure(figsize=(30, 15))
        ax = gF.addAxis(fig, 1, 2)
        ax[0].plot(self.loss_vector)
        ax[1].plot(np.log10(self.loss_vector))
        gF.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss")
        gF.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Log Loss")
        plt.savefig(self.savepath + "trainingOutput/png_images/trainingHistory.png")
        plt.savefig(self.savepath + "trainingOutput/pdf_images/trainingHistory.pdf")
        plt.close()

    def customLoad(self):
        # If a checkpoint file exists then load the checkpoint weights to architecture
        print("Checking for model checkpoint at: " + self.savepath)
        if os.path.exists(self.savepath + "checkpoint"):
            self.load_weights(self.savepath).expect_partial()
            print("\n Model Checkpoint Loaded \n")

        # Load the previous training loss vector if it exists
        if os.path.exists(self.savepath + "trainingHistory.pickle"):
            with open(self.savepath + "trainingHistory.pickle", "rb") as handle:
                trackHistory = pickle.load(handle)
                self.loss_vector = trackHistory["trainingLoss"]

    def visualizeTrainingCheckpoint(self, epoch_str):
        # It is expected that this function is overloaded by child class
        return

    def get_trainable_variables(self):
        return self.trainable_variables

    def get_variable_names(self):
        trianable_variables = self.get_trainable_variables()
        variable_names = [v.name for v in trianable_variables]
        return variable_names

    def get_variable_by_name(self, name):
        # Note that variables have str like ':0' at the end depending on the operation number
        if not ":" in name:
            name = name + ":0"

        return [v for v in self.trainable_variables if v.name == name]


# # EXAMPLE/COMMON PIPELINES PRE-IMPLEMENTED
# class pipeline_modulation_psf(pipeline_Object):
#     """Pipeline to optimize one (or more) continous phase profiles with a PSF objective function. Returns the batched,
#     stacked computed PSFs for the single-layer phase optics.

#     Attributes:
#         `prop_parameters` (prop_params): Propagation parameters object defining the fourier simulation settings. May
#             use "wavelength_m" or "wavelength_set_m" key for single wavelength or broadband simulation.
#         `point_source_locs` (tf.float): Coordinates of point-sources to compute the PSF for, of shape (N,3).
#         `psf_layer` (DFlat layer): The PSF computation instance layer used in the call function.
#         `trans` (tf.Variable): Transmission tensor, set to non-trainable, of shape
#             (num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or (num_profiles, 1, ms_samplesM["r"]).
#         `phase` (tf.Variable): Phase tensor, set to trainable, same shape as trans
#     """

#     def __init__(self, prop_parameters, point_source_locs, num_profiles, savepath, saveAtEpochs=None):
#         """Initializes the pipeline object.

#         Args:
#             `prop_parameters` (prop_params): Propagation parameters object defining the fourier simulation settings. May
#                 use "wavelength_m" or "wavelength_set_m" key for single wavelength or broadband simulation.
#             `point_source_locs` (float): Coordinates of point-sources to compute the PSF for, of shape (N,3).
#             `num_profiles` (int): Number of phase profiles to batch compute the psf over and optimize for.
#             `savepath` (str): Save path to create a folder and save training checkpoints to.
#             `saveAtEpochs` (int): Number of training epochs to save an intermediary model checkpoint.
#         """
#         super(pipeline_modulation_psf, self).__init__(savepath, saveAtEpochs)

#         self.prop_parameters = prop_parameters
#         self.point_source_locs = tf.convert_to_tensor(point_source_locs, prop_parameters["dtype"])
#         self.__radial_flag = prop_parameters["radial_symmetry"]

#         # Define the optimization pipeline layers
#         # NOTE: One could just use psf_broadband_layer always with len(wavelength_set_m)=1 for the single wavelength case
#         # but calling the explicit single wavelength code directly may present some computational speedup during training.
#         broadband_flag = prop_parameters["broadband_flag"]
#         if broadband_flag:
#             self.psf_layer = psf_broadband_layer(prop_parameters)
#         else:
#             self.psf_layer = psf_layer(prop_parameters)

#         # Initialize trainable phase and transmittance profile
#         ms_samplesM = prop_parameters["ms_samplesM"]
#         if prop_parameters["radial_symmetry"]:
#             profile_shape = [num_profiles, 1, ms_samplesM["r"]]
#         else:
#             profile_shape = [num_profiles, ms_samplesM["y"], ms_samplesM["x"]]

#         uniform_init = tf.ones(shape=profile_shape, dtype=prop_parameters["dtype"])
#         self.trans = tf.Variable(uniform_init, trainable=False, name="trans")
#         self.phase = tf.Variable(uniform_init, trainable=True, name="phase")

#     def __call__(self):
#         """Pipeline call function.

#         Returns:
#             `list`: List containing the PSF stack intensity in the first argument and the PSF phase in the second.
#                 Shape is (1 or len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"]).
#         """
#         return self.psf_layer(self.trans, self.phase, self.point_source_locs)

#     def visualizeTrainingCheckpoint(self, epoch_str):
#         intensity, _ = self.psf_layer(self.trans, self.phase, self.point_source_locs)
#         intensity = np.sum(intensity, axis=(0, 2))

#         fig = plt.figure(figsize=(30, 30))
#         ax = gF.addAxis(fig, 2, 2)
#         if self.__radial_flag:
#             ax[0].plot(self.phase[0, 0, :])
#             ax[1].plot(self.phase[1, 0, :])
#         else:
#             ax[0].imshow(self.phase[0, :, :])
#             ax[1].imshow(self.phase[1, :, :])
#         ax[2].imshow(intensity[0, :, :])
#         ax[3].imshow(intensity[1, :, :])
#         plt.savefig(self.savepath + "trainingOutput/png_images/Phase_Intensity_Epoch" + epoch_str + ".png")
#         plt.close()

#         return

#     def initialize_starting_profile(self, trans, phase):
#         """ Update the initial states for the tf Variables to the passed in trans and phase profiles.

#         Args:
#             trans: transmittance profile to use (note transmittance variable is not trainable)
#             phase: phase profile to use, of shape (num_profiles, 1, ms_samplesM["r"]) or
#                 (num_profiles, ms_samplesM["y"], ms_samplesM["x"])
#         """
#         self.trans = tf.Variable(trans, trainable=False, name="trans")
#         self.phase = tf.Variable(phase, trainable=True, name="phase")

#         return


# class pipeline_latent_mlp_psf(pipeline_Object):
#     """Pipeline to optimize a metasurface with a PSF objective function, using the latent neural optical representation
#     for metasurface cells. Returns the batch stacked computed PSFs for the single-layer metasurface.

#     Attributes:
#         `prop_parameters` (prop_params): Broadband propagation parameters, with the "wavelength_set_m" key defined.
#         'point_source_locs' (tf.float): List of point-source coordinates to compute the PSF for, of shape (N,3).
#         `psf_layer` (DFlat Layer): PSF compute instance used to calculate the measured PSF at the sensor.
#         `mlp_layer` (DFlat Layer): Neural optical model instance used to convert cell shapes to optical phase and trans.
#         `wavelength_m_asList` (list): list of wavelengths that the PSF is batch computed over.
#         `grid_shape` (list): Metasurface grid shape.
#         `latent_tensor` (tf.Variable): Trainable tensorflow variable containing the current normalized shape parameters
#             for the cells of each grid pixel.
#     """

#     def __init__(self, mlp_model_name, prop_parameters, point_source_locs, savepath, saveAtEpochs=None):
#         """Initialize the  latent-mlp-psf pipeline.

#         Args:
#             `mlp_model_name` (str): Name of the mlp model to use for shape optimization. See technical documentation for options.
#             `prop_parameters` (prop_params): Propagation parameters dictionary describing the lens to detector propagation.
#             `point_source_locs` (float): Point-source coordinates to batch compute the PSFs for, of shape (N,3).
#             `savepath` (str): Save path to create a folder and save training checkpoints to.
#             `saveAtEpochs` (int): Number of training epochs to save an intermediary model checkpoint.
#         """
#         super(pipeline_latent_mlp_psf, self).__init__(savepath, saveAtEpochs)

#         self.prop_parameters = prop_parameters
#         self.point_source_locs = tf.convert_to_tensor(point_source_locs, prop_parameters["dtype"])
#         self.__radial_flag = prop_parameters["radial_symmetry"]

#         ### Initialize computational layers
#         self.psf_layer = psf_broadband_layer(prop_parameters)
#         self.wavelength_m_asList = prop_parameters["wavelength_set_m"]
#         self.mlp_layer = mlp_latent_layer(mlp_model_name)

#         ### Define the grid shape, required for reshaping MLP output
#         ms_samplesM = prop_parameters["ms_samplesM"]
#         if self.__radial_flag:
#             self.grid_shape = [1, 1, ms_samplesM["r"]]
#         else:
#             self.grid_shape = [1, ms_samplesM["y"], ms_samplesM["x"]]

#         ### Define a trainable latent tensor
#         uniform_init = self.mlp_layer.initialize_input_tensor("uniform", self.grid_shape)
#         self.latent_tensor = tf.Variable(uniform_init, trainable=True)

#     def __call__(self):
#         """Pipeline call function

#         Returns:
#             `list`: List containing the PSF stack intensity in the first argument and the PSF phase in the second.
#                 Shape is (len(wavelength_m_asList), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"]).
#         """
#         trans, phase = self.mlp_layer(self.latent_tensor, self.wavelength_m_asList, self.grid_shape)
#         return self.psf_layer(trans, phase, self.point_source_locs)

#     def visualizeTrainingCheckpoint(self, epoch_str):
#         # Get the computed psf intensity; For plot, sum over wavelength and depth
#         trans, phase = self.mlp_layer(self.latent_tensor, self.wavelength_m_asList, self.grid_shape)
#         intensity, _ = self.psf_layer(trans, phase, self.point_source_locs)
#         intensity = np.sum(intensity.numpy(), axis=(0, 2))

#         # Get the shape parameters
#         shape_dist = self.mlp_layer.latent_to_unnorm_shape(self.latent_tensor, self.grid_shape)

#         # Plot Detector intensity and the metasurface distribution
#         if self.__radial_flag:
#             fig = plt.figure(figsize=(30, 30))
#             ax = gF.addAxis(fig, 2, 2)
#             ax[0].plot(shape_dist[0, 0, :, 0])
#             ax[1].plot(shape_dist[0, 0, :, 1])
#             ax[2].imshow(intensity[0])
#             ax[3].imshow(intensity[1])
#         else:
#             fig = plt.figure(figsize=(30, 30))
#             ax = gF.addAxis(fig, 2, 2)
#             im0 = ax[0].imshow(shape_dist[0, :, :, 0])
#             gF.formatPlots(fig, ax[0], im0, addcolorbar=True)
#             im1 = ax[1].imshow(shape_dist[0, :, :, 1])
#             gF.formatPlots(fig, ax[1], im1, addcolorbar=True)
#             ax[2].imshow(intensity[0])
#             ax[3].imshow(intensity[1])

#         plt.savefig(self.savepath + "trainingOutput/png_images/Meta_Intensity" + epoch_str + ".png")
#         plt.close()

#         return


# class pipeline_latent_rcwa_psf(pipeline_Object):
#     """Pipeline to optimize a metasurface given a PSF objective function, using the physical rcwa solver for
#     metasurface cells.

#     Attributes:
#         `rcwa_parameters` (rcwa_params): RCWA configuration dictionary, defining the rcwa_layer settings.
#         `prop_parameters` (prop_params): Propagation parameters dictionary describing the lens to detector propagation.
#         `point_source_locs` (tf.float): Point-source coordinates to batch compute the PSFs for, of shape (N,3).
#         `psf_layer` (Dflat Layer): The PSF computation instance layer used in the call function.
#         `rcwa_layer` (Dflat Layer): The RCWA physical optical model layer used in the call function.
#         `wavelength_m_asList` (float): List of wavelengths to batch compute the psf for, in units of meters.
#         `latent_tensor` (tf.Variable): Trainable tensorflow variable containing the current state normalized shape parameters
#             for the cells in each grid pixel.
#     """

#     def __init__(self, rcwa_parameters, prop_parameters, point_source_locs, savepath, saveAtEpochs=None):
#         """Initialize the latent-rcwa-psf pipeline.

#         Args:
#             `rcwa_parameters` (rcwa_params): RCWA configuration dictionary, defining the rcwa_layer settings.
#             `prop_parameters` (prop_params): Propagation parameters dictionary describing the lens to detector propagation.
#             `point_source_locs` (float): Point-source coordinates to batch compute the PSFs for, of shape (N,3).
#             `savepath` (str): Save path to create a folder and save training checkpoints to.
#             `saveAtEpochs` (int): Number of training epochs to save an intermediary model checkpoint.
#         """
#         super(pipeline_latent_rcwa_psf, self).__init__(savepath, saveAtEpochs)
#         self.rcwa_parameters = rcwa_parameters
#         self.prop_parameters = prop_parameters
#         self.point_source_locs = tf.convert_to_tensor(point_source_locs, prop_parameters["dtype"])
#         self.__radial_flag = prop_parameters["radial_symmetry"]

#         ### Initialize computational layers
#         # Initialize the psf_layer
#         self.psf_layer = psf_broadband_layer(prop_parameters)
#         self.wavelength_m_asList = prop_parameters["wavelength_set_m"]

#         # Initialize the RCWA layer using the rcwa_params object
#         self.rcwa_layer = rcwa_latent_layer(rcwa_parameters)
#         shape_vect_size = self.rcwa_layer.shape_vect_size

#         ### Define a trainable latent tensor
#         uniform_init = 0.5 * tf.ones(shape=shape_vect_size, dtype=tf.float32)
#         self.latent_tensor = tf.Variable(uniform_init, trainable=True)

#     def __call__(self):
#         """Pipeline call function

#         Returns:
#             `list`: List containing the PSF stack intensity in the first argument and the PSF phase in the second.
#                 Shape is (len(wavelength_m_asList), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"]).
#         """
#         trans, phase = self.rcwa_layer(self.latent_tensor)
#         trans = tf.cast(trans, dtype=tf.float64)
#         phase = tf.cast(phase, dtype=tf.float64)
#         return self.psf_layer(trans, phase, self.point_source_locs)

#     def visualizeTrainingCheckpoint(self, saveTo):
#         #
#         return

#     def initialize_starting_profile(self):
#         #
#         return
