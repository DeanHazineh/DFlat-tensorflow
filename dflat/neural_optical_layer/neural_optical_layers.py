import tensorflow as tf
import numpy as np
from .core.mlp_Dense_models import *
from .core.eRBF_models import *
from .core.mlp_call_helper import *
from dflat.tools.latent_param_utils import latent_to_param, param_to_latent

listModelNames = erbf_model_names + mlp_model_names


class MLP_Latent_Layer(tf.keras.layers.Layer):
    """Neural-Optical Cell Model Layer; Initialized to call one of D-Flats pre-trained MLPs. This layer computes the
    optical modulation (zero-order transmittance and phase) for cells, at user requested wavelengths, given a latent
    vector input. For input of the normalized parameters, rather than the latent vector, use MLP_Layer instead.

    Once initialized with a MLP selection, this class may be recalled to evaluate different latent tensors.

    Attributes:
        `mlp` (tf.keras.Model): MLP object/class initialized in the layer
        `mlp_input_shape` (int): Input mlp shape of (1,D+1), where D is the shape degree and an extra column
            specifies wavelength.
        `output_stack_dim` (int): MLP trans and phase output batch size (==1 for polarization insensitive or ==2 for
            polarization sensitive optics)
    """

    def __init__(self, model_name):
        """Initialize the mlp_latent_layer.

        Args:
            `model_name` (str): Name of the MLP model to use. See up-to-date documentation for valid models.
        """
        super(MLP_Latent_Layer, self).__init__()

        # Check model request and initialize the chosen model
        # Model weights are set to non-trainable for inference by default
        self.mlp = self.__init_MLP_model(model_name)

        # Get mlp input/output sizes
        input_shape_tuple = self.mlp.get_input_shape()
        self.mlp_input_shape = input_shape_tuple[0]
        self.output_stack_dim = self.mlp.get_output_pol_state()

    def __call__(self, latent_tensor, wavelength_m_asList):
        """Call function for the mlp_latent_layer. Given a latent tensor containing the transformed shape parameters
        for each cell and a list of wavelengths to evaluate the optical response for, the MLP predicted phase and
        transmittance is returned.

        Args:
            `latent_tensor` (tf.float): Tensor of cells shape parameters converted to latent space (see technical
                documents and reference paper), of shape (D, PixelsY, PixelsX) where D is the shape degree.
            `wavelength_m_asList` (list): List of wavelengths (in units of meters!) to evaluate the metasurface
                structure's optical response at.

        Returns:
            `list`: List containing transmittance in the first argument and phase in the second, of shape
                (len(wavelength_m_asList), p, PixelsY, PixelsX), where p = 1 or 2 depending on the model.
        """
        tf.debugging.assert_equal(
            tf.shape(latent_tensor).shape,
            3,
            message="latent_tensor should be a rank 3 tensor",
            name="latent_tensor_rank_assertion",
        )
        tf.debugging.assert_equal(
            tf.shape(latent_tensor)[0],
            self.mlp_input_shape - 1,
            message="",
            summarize="latent_tensor has incorrect number of columns (shape parameters D)",
            name="latent_tensor_degree_assertion",
        )

        # Convert a latent input to normalized parameters and in suitable form for MLP, (MxD).
        gridShape = [1, latent_tensor.shape[1], latent_tensor.shape[2]]
        latent_tensor = flatten_reshape_shape_parameters(latent_tensor)
        norm_param = latent_to_param(latent_tensor)

        return batched_broadband_MLP(norm_param, self.mlp, wavelength_m_asList, gridShape, self.output_stack_dim,)

    def __init_MLP_model(self, model_selection_string):
        if model_selection_string not in listModelNames:
            raise ValueError("mlp_layer: requested MLP is not one of the supported libraries")
        else:
            mlp = globals()[model_selection_string]
            mlp = mlp()
            mlp.customLoadCheckpoint()
            mlp.trainable = False

        return mlp

    def initialize_input_tensor(self, init_type, gridShape, dtype=tf.float64, init_args=[]):
        """Initialize a latent_tensor input. Valid initializations here are "uniform" and "random". To use an
            alternative, user-defined starting latent_tensor, one may be able to create their own
            using mlp_initialization_utilities.optical_response_to_param and a suitable param_to_latent call.

        Args:
            `init_type` (str): Selection of initialization types, either "uniform", "random"
            `gridShape` (list): 2D cell grid shape given as a length three list, usually of the form [1, ms_samplesM["y"], ms_samples["x"]] or [1, 1, ms_samples["r"]].
            `dtype` (tf.dtype, optional): Data-type for the returned tensor. Defaults to tf.float64

        Returns:
            `dtype`: Latent_tensor of suitable form to pass to mlp_latent_layer call function.
        """
        norm_param = init_norm_param(init_type, dtype, gridShape, self.mlp_input_shape, init_args)
        return param_to_latent(norm_param)

    def latent_to_unnorm_shape(self, latent_tensor, gridShape=None):
        """Given a latent_tensor, this function returns the unnormalized shape distributions matching the physical
        metasurface implementation, with an option of reshaping.

        Args:
            `latent_tensor` (tf.float): Tensorflow tensor containing the latent-space shape parameters.
            `gridShape` (list, optional): The dimensions to reshape the latent vector to. Defaults to None.

        Returns:
            `np.array`: Unnormalized shape parameters, as a list of cells (M, D) or as a reshaped as (gridShape,D).
        """
        norm_param = latent_to_param(latent_tensor)
        unnorm_shape = unnormalize_shapeVector_np(norm_param.numpy(), self.mlp)
        if gridShape:
            unnorm_shape = np.reshape(unnorm_shape, np.append(gridShape, self.mlp_input_shape - 1))

        return unnorm_shape


class MLP_Layer(MLP_Latent_Layer):
    """Neural-Optical Cell Model Layer; Initialized to call one of D-Flats pre-trained MLPs. This layer computes the
    optical modulation (zero-order transmittance and phase) for cells, at user requested wavelengths, given a
    normalized shape tensor. For latent parameter input, use mlp_latent_layer instead.

    Attributes:
        `mlp` (tf.keras.Model): MLP object/class initialized in the layer
        `mlp_input_shape` (int): Input mlp shape degree D, such that MLP takes input of shape (1,D+1), where +1 is due
             to the addition of wavelength state.
        `output_stack_dim` (int): MLP trans and phase output batch size (==1 for polarization insensitive or ==2
            for polarization sensitive optics)
    """

    def __init__(self, model_name):
        super(MLP_Layer, self).__init__(model_name)
        """Initialize the mlp_latent_layer. 

        Args:
            `model_name` (str): Name of the MLP model to use. See up-to-date documentation for valid models. 
        """

    def __call__(self, norm_param, wavelength_m_asList):
        """Call function for the mlp_layer. Given a normalized shape vector containing the shape parameters for each
        cell and a list of wavelengths to evaluate the optical response for, the MLP predicted phase and
        transmittance for each wavelength channel is returned.

        Args:
            `norm_param` (tf.float): Tensor of cells normalized shape parameters (see technical documents and
                reference paper), of shape (D, PixelsY, PixelsX) where D is the shape degree.
            `wavelength_m_asList` (list): List of wavelengths (in units of meters!) to evaluate the metasurface
                structure's optical response at.

        Returns:
            `list`: List containing transmittance in the first argument and phase in the second, of shape
                (len(wavelength_m_asList), p, gridShape[-2], gridShape[-1]), where p = 1 or 2 depending on the model.
        """
        # For now, we want to manually assert input is the correct shape
        tf.debugging.assert_equal(
            tf.shape(norm_param).shape,
            3,
            message="norm_param should be a rank 2 tensor",
            name="norm_param_rank_assertion",
        )
        tf.debugging.assert_equal(
            tf.shape(norm_param)[0],
            self.mlp_input_shape - 1,
            message="",
            summarize="norm_param has incorrect number of columns (shape parameters D)",
            name="norm_param_degree_assertion",
        )

        gridShape = [1, norm_param.shape[1], norm_param.shape[2]]
        norm_param = flatten_reshape_shape_parameters(norm_param)

        return batched_broadband_MLP(norm_param, self.mlp, wavelength_m_asList, gridShape, self.output_stack_dim,)

    def initialize_input_tensor(self, init_type, gridShape, dtype=tf.float64, init_args=[]):
        """Initialize a normalized shape param input. Valid initializations here are "uniform" and "random". To use an
        alternative, user-defined starting param tensor, one may be able to create their own using
        mlp_initialization_utilities.optical_response_to_param.

        Args:
            `init_type` (str): Selection of initialization types, either "uniform" or "random"
            `gridShape` (list): 2D cell grid shape given as a len=3 list, usually of the form
                [1, ms_samplesM["y"], ms_samples["x"]] or [1, 1, ms_samples["r"]].
            `dtype` (tf.dtype, optional): Data-type for the returned tensor. Defaults to tf.float64

        Returns:
            dtype: norm_param tensor of suitable form to pass to mlp_latent_layer call function.
        """
        return init_norm_param(init_type, dtype, gridShape, self.mlp_input_shape, init_args)

    def param_to_unnorm_shape(self, norm_param, gridShape=None):
        """Given the normalized_parameters, this function returns the unnormalized shape distributions matching the physical
        metasurface implementation, with an option of reshaping.

        Args:
            `norm_param` (tf.float): Tensorflow tensor containing the normalized shape parameters.
            `gridShape` (list, optional): The dimensions to reshape the parameter list to. Defaults to None.

        Returns:
            `np.array`: Unnormalized shape parameters, as a list of cells (M, D) or as a reshaped as (gridShape,D).
        """
        unnorm_shape = unnormalize_shapeVector_np(norm_param.numpy(), self.mlp)
        if gridShape:
            unnorm_shape = np.reshape(unnorm_shape, np.append(gridShape, self.mlp_input_shape - 1))

        return unnorm_shape
