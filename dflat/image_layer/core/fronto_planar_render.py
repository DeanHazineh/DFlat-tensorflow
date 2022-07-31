import tensorflow as tf
from .fourier_convolve import *


def render_depth_naive(psf_intensity, pinhole_image, seg_mask):
    return tf.math.reduce_sum(fourier_convolve_real(pinhole_image * seg_mask, psf_intensity), axis=[1, 3])


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
