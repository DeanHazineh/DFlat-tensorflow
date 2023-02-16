import tensorflow as tf
import numpy as np

from .core.ops_fft_convolve import general_convolve
from .core.ops_measurement import photons_to_ADU

# from .core.fronto_planar_render import *
# from .core.sensor_functions import *

# PSF (broadband) will have shapes: (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
# PSF (monochromatic case) will have shape: (profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"]).


class Fronto_Planar_renderer_incoherent(tf.keras.layers.Layer):
    def __init__(self, sensor_parameters):
        super(Fronto_Planar_renderer_incoherent, self).__init__()

        # ### Add sensor_parameters to class attributes
        # for key, value in sensor_parameters.items():
        #     self.__dict__[key] = value
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


# class Fronto_Planar_HS_to_RGB(tf.keras.layers.Layer):
#     def __init__(self, wavelength_set_m, dtype):
#         super(Fronto_Planar_HS_to_RGB, self).__init__()

#         # For RGB photosensor, get the sensors transmission Filters (with QE included)
#         self.wavelength_set_m = wavelength_set_m
#         sensorQE, column_labels = get_QETrans_Basler_Bayer(wavelength_set_m * 1e9)
#         self.sensorQE = tf.convert_to_tensor(sensorQE, dtype=dtype)

#     def __call__(self, psf_intensity, hsi_image_batch):
#         # PSF_Intensity of shape (Nwl, Nlens, Nps, Ny, Nx)
#         # Images passed in shape (Nbatch, H, W, Nwl)

#         # Check on the input shapes
#         if tf.shape(psf_intensity).shape != 5:
#             raise ValueError("psf_intensity should be a rank 5 tensor. Check docstrings")

#         if tf.shape(hsi_image_batch).shape != 4:
#             raise ValueError("hsi_image_batch should be a rank 4 tensor. Check docstrings")

#         if tf.shape(psf_intensity)[0] != tf.shape(hsi_image_batch)[3] != len(self.wavelength_set_m):
#             raise ValueError("Wavelenght channels should be the same for the PSF, hsi_images, and wavelength vector")

#         # # Check if tesnors
#         # # What should I do about datatyping?
#         # if not tf.is_tensor(psf_intensity):
#         #     psf_intensity = tf.convert_to_tensor(psf_intensity)
#         # if not tf.is_tensor(hsi_image_batch):
#         #     hsi_image_batch = tf.convert_to_tensor(hsi_image_batch)

#         rgb = render_spectral_to_rgb_image(psf_intensity, hsi_image_batch, self.wavelength_set_m)
#         mono = render_spectral_to_monochrome(psf_intensity, hsi_image_batch)
#         return mono, rgb


# class Fronto_Planar_Renderer_Sparse(tf.keras.layers.Layer):
#     """_summary_

#     Args:
#         tf (_type_): _description_
#     """

#     def __init__(
#         self,
#     ):
#         """_summary_"""
#         super(Fronto_Planar_Renderer_Sparse, self).__init__()

#     def __call__(self, psf_intensity, focus_image, seg_mask):
#         """Renders the detector image given, a PSF, segmentation_maks, and corresponding PSFs. Currently, this function
#         only incorporates the most basic, naive, depth rendering image formation model.

#         Args:
#             'psf_intensity' (tf.float): PSF for rendering, of shape (batch or None, N_lambda, N_b, N_z, Ny, Nx).
#             'focus_image' (tf.float): Pinhole image of shape (batch or None, N_lambda, N_b, 1, Ny, Nx), dimension broadcast allowed.
#             'seg_mask' (tf.float): Segmentation masks for scene, of shape (batch or None, , 1, 1, N_z, Ny, Nx)

#         Returns:
#             tf.float: Rendered sensor image of shape, (N_b, Ny, Nx)
#         """

#         # Ensure tensor inputs
#         dtype = psf_intensity.dtype
#         if not tf.is_tensor(focus_image):
#             focus_image = tf.convert_to_tensor(focus_image, dtype=dtype)

#         if not tf.is_tensor(psf_intensity):
#             psf_intensity = tf.convert_to_tensor(psf_intensity, dtype=dtype)

#         if not tf.is_tensor(seg_mask):
#             seg_mask = tf.convert_to_tensor(seg_mask, dtype=dtype)

#         return render_depth_naive(psf_intensity, focus_image, seg_mask)
