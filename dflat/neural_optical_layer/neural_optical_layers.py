import tensorflow as tf

from dflat.tools.core.latent_param_utils import latent_to_param, param_to_latent
from .core.neural_model_util import *


class MLP_Layer(tf.keras.layers.Layer):
    """Neural-Optical Cell Model Layer; Initialized to call one of D-Flats pre-trained MLPs. This layer computes the
    optical modulation (zero-order transmittance and phase) for cells, at user requested wavelengths, given a (normalized) parameter
    input. For input of the latent vector, rather than the parameter vector, use MLP_Latent_Layer instead.

    Once initialized with a MLP selection, this class may be recalled to evaluate different parameter tensors.

    Attributes:
        `mlp` (tf.keras.Model): MLP object/class initialized in the layer
        `mlp_input_shape` (int): Input mlp shape of (1,D+1), where D is the shape degree and an extra column
            specifies wavelength.
    """

    def __init__(self, model_name, dtype=tf.float64):
        """Initialize the mlp_layer.
        Args:
            `model_name` (str): Name of the MLP model to use. See up-to-date documentation for valid models.
        """
        super(MLP_Layer, self).__init__()

        # Check model request and initialize the chosen model - Model weights are set to non-trainable for inference by default
        self.mlp = load_neuralModel(model_name, dtype)
        self._dtype = dtype
        self.__check_input_shape = False
        self.__check_input_type = False
        self.input_dimensionality = self.mlp.get_input_shape()
        self.param_dimensionality = self.input_dimensionality - 1

    def __call__(self, norm_param, wavelength_m_asList):
        """Call function for the mlp_layer. Given a normalized parameter vector for each
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

        # Assert input shape
        if not self.__check_input_shape:
            self.check_shape(norm_param)
        if not self.__check_input_type:
            norm_param = self.check_dtype(norm_param)

        # norm_params passed to MLP need to be reshaped into [N, D]
        gridShape = [1, norm_param.shape[1], norm_param.shape[2]]
        norm_param = flatten_reshape_shape_parameters(norm_param)

        return batched_broadband_MLP(
            norm_param,
            self.mlp,
            wavelength_m_asList,
            gridShape,
        )

    def check_shape(self, input_tensor):
        if input_tensor.ndim != 3:
            raise ValueError("norm_param should be a rank 3 tensor: (D, PixelsY, PixelsX)")

        if input_tensor.shape[0] != self.param_dimensionality:
            raise ValueError("norm param has unexpected dimensionality. In this case it should be: ", self.param_dimensionality)

        self.__check_input_shape = True
        return

    def check_dtype(self, input_tensor):
        if not tf.is_tensor(input_tensor):
            input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        elif input_tensor.dtype != tf.float32:
            input_tensor = tf.cast(input_tensor, tf.float32)
        else:
            self.__check_input_type = True

        return input_tensor

    def initialize_input_tensor(self, init_type, gridShape, init_args=[]):
        """Initialize a normalized param input. Valid initializations here are "uniform" and "random".

        To use an alternative, user-defined starting param tensor, one cam define another function in range [0,1] or create
        via phase/transmission to shape lookup with dflat.datasets_metasurface_cells.optical_response_to_param if a library exists

        Args:
            `init_type` (str): Selection of initialization types, either "uniform" or "random"
            `gridShape` (list): 2D cell grid shape given as a rank 3 array, usually of the form
                [1, ms_samplesM["y"], ms_samples["x"]] or [1, 1, ms_samples["r"]].

        Returns:
            dtype: norm_param tensor of suitable form to pass to mlp_latent_layer call function.
        """
        return init_norm_param(init_type, self._dtype, gridShape, self.input_dimensionality, init_args)

    def param_to_shape(self, norm_param):
        """Given the normalized_parameters, this function returns the unnormalized shape distributions matching the physical
        metasurface implementation, with an option of reshaping.
        Args:
            `norm_param` (tf.float): Tensor of cells normalized parameters of shape (D, PixelsY, PixelsX) where D is the shape degree.
        Returns:
            `tf.float`: Unnormalized shape parameters, same shape as norm_param input.
        """
        # input should be reshaped like the MLP input shape, [N, D] before calling helper function
        gridShape = norm_param.shape
        norm_param = flatten_reshape_shape_parameters(norm_param)
        shape_vector = convert_param_to_shape(norm_param, self.mlp)
        return tf.transpose(tf.reshape(shape_vector, [gridShape[1], gridShape[2], -1]), [2, 0, 1])

    def shape_to_param(self, shape_vect):
        """Given the shape vector (units of m), this function returns the normalized parameter vector for the mlp model
        Args:
            `shape_vect` (tf.float): Tensor of cells shape parameters (D, PIxelsY, PixelsX) where D is the shape degree.
        Returns:
            `tf.float`: Normalized shape parameters, same size as shape_vect
        """

        # input should be reshaped like the MLP input shape, [N, D] before calling helper function
        gridShape = shape_vect.shape
        shape_vect = flatten_reshape_shape_parameters(shape_vect)
        norm_param = convert_shape_to_param(shape_vect, self.mlp)
        return tf.transpose(tf.reshape(norm_param, [gridShape[1], gridShape[2], -1]), [2, 0, 1])


class MLP_Latent_Layer(MLP_Layer):
    """Neural-Optical Cell Model Layer; Initialized to call one of D-Flats pre-trained MLPs. This layer computes the
    optical modulation (zero-order transmittance and phase) for cells, at user requested wavelengths, given a latent
    vector input. For input of the normalized parameters, rather than the latent vector, use MLP_Layer instead.

    Once initialized with a MLP selection, this class may be recalled to evaluate different latent tensors.

    Attributes:
        `mlp` (tf.keras.Model): MLP object/class initialized in the layer
        `mlp_input_shape` (int): Input mlp shape of (1,D+1), where D is the shape degree and an extra column
            specifies wavelength.
    """

    def __init__(self, model_name, pmin=0, pmax=1, dtype=tf.float64):
        """Initialize the mlp_latent_layer.

        Args:
            `model_name` (str): Name of the MLP model to use. See up-to-date documentation for valid models.
            `pmin` (tf.float): minimum value for the normalized parameter, in range [0, 1)
            `pmax` (tf.float): maximum value for the normalized parameter, in range (pmin, 1]
        """
        super(MLP_Latent_Layer, self).__init__(model_name, dtype)
        self.pmin = pmin
        self.pmax = pmax

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

        # Assert inputs
        if not self.__check_input_shape:
            self.check_shape(latent_tensor)
        if not self.__check_input_type:
            latent_tensor = self.check_dtype(latent_tensor)

        # Ensure input is float32 to match the neural network
        # The output of the model will be cast to self.dtype
        if not tf.is_tensor(latent_tensor):
            latent_tensor = tf.convert_to_tensor(latent_tensor, dtype=tf.float32)
        elif latent_tensor.dtype != tf.float32:
            latent_tensor = tf.cast(latent_tensor, tf.float32)

        # params passed to MLP need to be reshaped into [N, D]
        norm_param = latent_to_param(flatten_reshape_shape_parameters(latent_tensor), self.pmin, self.pmax)
        gridShape = [1, latent_tensor.shape[1], latent_tensor.shape[2]]

        return batched_broadband_MLP(norm_param, self.mlp, wavelength_m_asList, gridShape)

    def initialize_input_tensor(self, init_type, gridShape, init_args=[]):
        """Initialize a latent_tensor input. Valid initializations here are "uniform" and "random". To use an
            alternative, user-defined starting latent_tensor, one may be able to create their own
            using mlp_initialization_utilities.optical_response_to_param and a suitable param_to_latent call.

        Args:
            `init_type` (str): Selection of initialization types, either "uniform", "random"
            `gridShape` (list): 2D cell grid shape given as a length three list, usually of the form [1, ms_samplesM["y"], ms_samples["x"]] or [1, 1, ms_samples["r"]].

        Returns:
            `dtype`: Latent_tensor of suitable form to pass to mlp_latent_layer call function.
        """

        norm_param = init_norm_param(init_type, self._dtype, gridShape, self.input_dimensionality, init_args)
        return param_to_latent(norm_param, self.pmin, self.pmax)

    def latent_to_shape(self, latent_tensor):
        """Given a latent_tensor, this function returns the unnormalized shape distributions matching the physical
        metasurface implementation, with an option of reshaping.

        Args:
            `latent_tensor` (tf.float): Tensorflow tensor containing the latent parameters, of shape (D, PixelsY, PixelsX) where D is the shape degree.

        Returns:
            `np.array`: Unnormalized shape parameters, same shape as input latent_tensor.
        """

        gridShape = latent_tensor.shape
        norm_param = flatten_reshape_shape_parameters(latent_to_param(latent_tensor, self.pmin, self.pmax))
        shape_vector = convert_param_to_shape(norm_param, self.mlp)

        return tf.transpose(tf.reshape(shape_vector, [gridShape[1], gridShape[2], -1]), [2, 0, 1])
