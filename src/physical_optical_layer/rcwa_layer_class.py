import tensorflow as tf
import numpy as np

from physical_optical_layer.core.batch_solver import full_rcwa_shape, batched_wavelength_rcwa_shape
from tools.latent_param_utils import latent_to_param, param_to_latent


class RCWA_Latent_Layer(tf.keras.layers.Layer):
    """RCWA_latent_layer; A physical optical model to evaluate the optical response of cells. 
    This layer computes the optical modulation (zero-order transmittance and phase) for cells, at batched wavelengths,
    given a latent vector input. For input of the normalized parameters, rather than the latent vector, use rcwa_layer
    instead. 
    
    Once initialized with a rcwa_parameters configuration, this class may be recalled to evaluate different latent tensors.

    Attributes:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solve settings
        `shape_vect_size` (list): Required shape for the input latent_vector, given the rcwa_params settings used 
            during layer initialization.
    """

    def __init__(self, rcwa_parameters):
        """Initialize the rcwa_latent_layer. 

        Args:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solver settings.
        """
        super(RCWA_Latent_Layer, self).__init__()

        self.rcwa_parameters = rcwa_parameters
        self.shape_vect_size = rcwa_parameters["shape_vect_size"]

        if rcwa_parameters["batch_wavelength_dim"]:
            self.rcwa_caller = batched_wavelength_rcwa_shape
        else:
            self.rcwa_caller = full_rcwa_shape

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
        return self.rcwa_caller(norm_param, self.rcwa_parameters)


class RCWA_Layer(RCWA_Latent_Layer):
    """RCWA_layer; A physical optical model to evaluate the optical response of cells. 
    This layer computes the optical modulation (zero-order transmittance and phase) for cells, at batched wavelengths, 
    given a shape parameter input. For input of the latent-space parameters, rather than the normalized shape parameters,
     use rcwa_latent_layer instead. 
    
    Once initialized with a rcwa_parameters configuration, this class may be recalled to evaluate different metasurfaces.

    Attributes:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solve settings
        `shape_vect_size` (list): Required shape for the input norm_param tensor, given the rcwa_params settings used 
            during layer initialization.
    """

    def __init__(self, rcwa_parameters):
        """Initialize the rcwa_layer. 

        Args:
        `rcwa_parameters` (rcwa_param): Configuration dictionary object providing the rcwa solver settings
        """
        super(RCWA_Layer, self).__init__(rcwa_parameters)

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
        tf.debugging.assert_equal(
            norm_param.shape,
            self.shape_vect_size,
            message="norm_param has incorrect size for the selected parameterization_type.",
            name="param_vector_shape_assertion",
        )

        return self.rcwa_caller(norm_param, self.rcwa_parameters)

