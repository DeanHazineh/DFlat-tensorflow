import tensorflow as tf
import numpy as np

from .core.ops_fft_convolve import general_convolve
from .core.ops_measurement import photons_to_ADU


class Fronto_Planar_renderer_incoherent(tf.keras.layers.Layer):
    def __init__(self, sensor_parameters):
        super(Fronto_Planar_renderer_incoherent, self).__init__()

        ### Add sensor_parameters to class attributes
        self.sensor_parameters = sensor_parameters

    def __call__(self, psf_intensity, AIF_image, collapse=False, rfft=False):
        """Renders image with appropriate PSF blurr for a fronto-planar scene components

        Args:
            'psf_intensity' (tf.float): PSF intensity of shape [Num_wl or None, num_profile, num_point_source, sensor_pix_y, sensor_pix_x]
            'AIF_image' (tf.float): All in focus images, the same rank as psf_intensity
            'collapse' (boolean; defaults False): If True, return image summed over the wavelength and point_source_channel
            'rfft' (boolean; defaults False): If True, compute transform via rfft instead of fft

        Returns:
            tf.float: rendered intensity image
        """

        ### Validate inputs
        init_rank = tf.rank(psf_intensity)
        psf_intensity, AIF_image = self.__check_inputs([psf_intensity, AIF_image])

        ### Compute the convolved image
        meas_photons = general_convolve(AIF_image, psf_intensity, rfft=rfft)

        ### Convert to digital, noisy measurement
        meas_adu = photons_to_ADU(meas_photons, self.sensor_parameters)

        # Return with or without collapse
        if collapse:
            return tf.math.reduce_sum(meas_adu, axis=[0, 2])
        elif init_rank == tf.TensorShape(4):
            return tf.squeeze(meas_adu, 0)
        else:
            return meas_adu

    def __check_inputs(self, tensor_args):
        out_arg = []
        for arg_input in tensor_args:
            # Ensure that the input is a tensor
            if not tf.is_tensor(arg_input):
                arg_input = tf.convert_to_tensor(arg_input)

            # For convenince, convert rank 4 input to rank 5 input
            if tf.rank(arg_input) == tf.TensorShape(4):
                arg_input = tf.expand_dims(arg_input, 0)
            elif not tf.rank(arg_input) == tf.TensorShape(5):
                raise ValueError("Input Tensor must be rank 4 or rank 5")

            out_arg.append(arg_input)

        return out_arg
