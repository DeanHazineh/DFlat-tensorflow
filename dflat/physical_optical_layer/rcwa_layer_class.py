import tensorflow as tf
from dflat.tools.core.latent_param_utils import latent_to_param, param_to_latent
from .core.batch_solver import batched_wavelength_rcwa_shape, compute_ref_field, full_rcwa_shape

from .core.ms_parameterization import ALLOWED_PARAMETERIZATION_TYPE, CELL_SHAPE_DEGREE


class RCWA_Layer(tf.keras.layers.Layer):
    """RCWA_layer; A physical optical model to evaluate the optical response of cells.
    This layer computes the optical modulation (zero-order transmittance and phase) for cells, at batched wavelengths,
    given a shape parameter input. For input of the latent-space parameters, rather than the normalized shape parameters,
    use rcwa_latent_layer instead.

    Once initialized with a rcwa_parameters configuration, this class may be recalled to evaluate different metasurfaces.

    Attributes:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solve settings
    """

    def __init__(self, rcwa_parameters, cell_parameterization, feature_layer=0):
        """Initialize the rcwa_layer.
        Args:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solver settings
        `cell_parameterization` (string): Cell parameterization model name
        `feature_layer` (int): Specify which layer of L the feature is to be placed in
        """

        super(RCWA_Layer, self).__init__()
        self.__check_parameterization_type(cell_parameterization)

        # Add class initializers to attributes
        self.rcwa_parameters = rcwa_parameters
        self.cell_parameterization = cell_parameterization
        self.shape_vect_size = self.__get_param_shape()
        self.feature_layer = feature_layer

        # Compute reference field and store it in attributes for use during call
        self.ref_field = compute_ref_field(rcwa_parameters)

        # batch wavelength will be deprecated  in future since its so inefficient, not worth it
        if rcwa_parameters["batch_wavelength_dim"]:
            self.rcwa_caller = batched_wavelength_rcwa_shape
        else:
            self.rcwa_caller = full_rcwa_shape

    def __call__(self, norm_param):
        """Call function for the rcwa_layer. Given a tensor containing the normalized shape parameters for each cell,
        the physical-model-predicted phase and transmission is returned.

        Args:
            `norm_param` (tf.float): Tensor of cell's normalized shape parameters (see technical documents and reference paper).
                The required shape can be obtained by calling class attribute self.shape_vect_size = (d1, PixelsX, PixelsY, d2),
                where d1 are shape parameters for each of the d2 number of structures placed in the cell

        Returns:
            `tf.float`: Transmittance stack, of shape (len(wavelength_m_asList), p, pixelsY, pixelsX), where p=2 for x and y polarizations.
            `tf.float`: Phase stack, of shape (len(wavelength_m_asList), p, pixelsY, pixelsX), where p=2 for x and y polarizations.

        """
        # Check that the norm_param vector passed in is the correct shape for the assembly type
        tf.debugging.assert_equal(
            norm_param.shape,
            self.shape_vect_size,
            message="norm_param has incorrect size for the selected parameterization_type.",
            name="param_vector_shape_assertion",
        )

        field = self.rcwa_caller(norm_param, self.rcwa_parameters, self.cell_parameterization, self.feature_layer)

        return tf.abs(field) / tf.abs(self.ref_field), tf.math.angle(self.ref_field) - tf.math.angle(field)

    def __check_parameterization_type(self, cell_param):
        if not (cell_param in ALLOWED_PARAMETERIZATION_TYPE.keys()):
            raise ValueError("Error in rcwa layer: parameterization_type not one of the allowed options")
        return

    def __get_param_shape(self):
        cell_shape_degree = CELL_SHAPE_DEGREE[self.cell_parameterization]
        rcwa_param = self.rcwa_parameters
        pixelsX = rcwa_param["pixelsX"]
        pixelsY = rcwa_param["pixelsY"]

        shape_vect_size = [cell_shape_degree[0], pixelsX, pixelsY, cell_shape_degree[1]]
        return shape_vect_size


class RCWA_Latent_Layer(RCWA_Layer):
    """RCWA_latent_layer; A physical optical model to evaluate the optical response of cells.
    This layer computes the optical modulation (zero-order transmittance and phase) for cells, allowing broadband wavelengths,
    given a latent vector input. For input of the normalized parameters, rather than the latent vector, use rcwa_layer
    instead.

    Once initialized with a rcwa_parameters configuration, this class may be recalled to evaluate different latent tensors.

    Attributes:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solve settings
        `shape_vect_size` (list): Required shape for the input latent_vector, given the rcwa_params settings used
            during layer initialization.
    """

    def __init__(self, rcwa_parameters, cell_parameterization, feature_layer=0):
        """Initialize the rcwa_latent_layer.

        Args:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solver settings.
        """
        super(RCWA_Latent_Layer, self).__init__(rcwa_parameters, cell_parameterization)

    def __call__(self, latent_vector):
        """Call function for the rcwa_latent_layer. Given a latent tensor containing the transformed shape parameters
        for each cell, the physical-model-predicted phase and transmittance is returned.

        Args:
            `latent_vector` (tf.float): Tensor of cell's shape parameters converted to latent space
                (see technical documents and reference paper). The required shape can be obtained by calling class
                attribute self.shape_vect_size = (d1, PixelsX, PixelsY, d2), where d1 are shape parameters for each of
                the d2 number of structures placed in the cell

        Returns:
            `tf.float`: Transmittance stack, of shape (len(wavelength_m_asList), p, pixelsY, pixelsX), where p=2 for x and y polarizations.
            `tf.float`: Phase stack, of shape (len(wavelength_m_asList), p, pixelsY, pixelsX), where p=2 for x and y polarizations.

        """
        tf.debugging.assert_equal(
            latent_vector.shape,
            self.shape_vect_size,
            message="latent_vector has incorrect size for the selected parameterization_type.",
            name="latent_vector_shape_assertion",
        )

        # Convert latent_vector to the normalized parameters
        norm_param = latent_to_param(latent_vector)
        field = self.rcwa_caller(norm_param, self.rcwa_parameters, self.cell_parameterization, self.feature_layer)

        return tf.abs(field) / tf.abs(self.ref_field), tf.math.angle(self.ref_field) - tf.math.angle(field)
