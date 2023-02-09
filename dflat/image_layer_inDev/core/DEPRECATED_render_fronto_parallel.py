import tensorflow as tf
from .ops_fft_convolve import *
from .DEPRECATED_tensor_spec_utils import hsi_to_rgb


# def render_spectral_to_rgb_image(psf_intensity, hsi_image_batch, wavelength_set_m):
#     # PSF_Intensity of shape (Nwl, Nlens, Nps, Ny, Nx)
#     # Images passed in shape (Nbatch, H, W, Nwl)

#     spec_im = render_spectral_image(psf_intensity, hsi_image_batch)  # [Num_batch, Nwl, Nlens, Nps, Ny, Nx]

#     spec_trans = tf.transpose(spec_im, [0, 2, 3, 4, 5, 1])  # move wavelength to the last dimension
#     return hsi_to_rgb(spec_trans, wavelength_set_m * 1e9)


# def render_spectral_to_monochrome(psf_intensity, hsi_image_batch):
#     return tf.math.reduce_sum(render_spectral_image(psf_intensity, hsi_image_batch), axis=[1, 3])


# def render_spectral_image(psf_intensity, hsi_image_batch):
#     # Convolve psf with image (e.g. as called in the Fronto_Planar_HS_to_RGB image layer)
#     # returns image in shape [Num_batch, Nwl, Nlens, Nps, Ny, Nx]

#     # cutting the PSF and removing energy is more problematic than arbitrary chops to the hsi image
#     # resize the hsi to the psf (at least for now)

#     # If the sensor sizes are different for the psf and hsi_image, make them the same (needed for fft_convolve)
#     psf_shape = tf.shape(psf_intensity)
#     hsi_shape = tf.shape(hsi_image_batch)
#     Py, Px = psf_shape[3], psf_shape[4]
#     Iy, Ix = hsi_shape[1], hsi_shape[2]
#     Ny = Py if Py > Iy else Iy
#     Nx = Px if Px > Ix else Ix

#     hsi_image_batch = tf.transpose(tf.image.resize_with_crop_or_pad(hsi_image_batch, Ny, Nx), [0, 3, 1, 2])
#     hsi_image_batch = hsi_image_batch[:, :, tf.newaxis, tf.newaxis, :, :]
#     psf_intensity = tf.image.resize_with_crop_or_pad(
#         tf.reshape(psf_intensity, [psf_shape[0] * psf_shape[1] * psf_shape[2], psf_shape[3], psf_shape[4], 1]), Ny, Nx
#     )
#     psf_rshape = psf_intensity.shape
#     psf_intensity = tf.reshape(psf_intensity, [psf_shape[0], psf_shape[1], psf_shape[2], psf_rshape[1], psf_rshape[2]])
#     psf_intensity = tf.expand_dims(psf_intensity, 0)  # expanded for batched images

#     print("TEST: ", psf_intensity.shape, hsi_image_batch.shape)

#     return fourier_convolve_real(hsi_image_batch, psf_intensity)


# def render_depth_naive(psf_intensity, pinhole_image, seg_mask):
#     return tf.math.reduce_sum(fourier_convolve_real(pinhole_image * seg_mask, psf_intensity), axis=[1, 3])


# def render_depth_naive_batched(psf_intensity, grayscale_im, seg_mask):
#     # PSF of shape [Nb, Nz, Ny, Nx]
#     num_batch = psf_intensity.shape[0]

#     # Use a tf while_loop to batch metasurfaces
#     def ms_loopCond(idx_, hold_im_):
#         return tf.less(idx_, num_batch)

#     def ms_loopBody(idx_, hold_im_):

#         im = render_depth_naive(psf_intensity[idx_], grayscale_im, seg_mask)
#         hold_im_ = tf.concat([hold_im_, tf.expand_dims(im, 0)], axis=0)
#         idx_ += 1
#         return [idx_, hold_im_]

#     # Run batched loop
#     dtype = psf_intensity.dtype
#     idx = tf.constant(0, dtype=tf.int32)
#     hold_im = tf.zeros((1, 1, psf_intensity.shape[-2], psf_intensity.shape[-1]), dtype=dtype)

#     loopData = tf.while_loop(ms_loopCond, ms_loopBody, loop_vars=[idx, hold_im])

#     return tf.expand_dims(tf.stack(loopData[1][1:]), 0)
