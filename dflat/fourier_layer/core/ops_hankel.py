import tensorflow as tf
import numpy as np
import scipy.special as scipy_bessel
from .ops_transform_util import *


# The QDHT code included here is a tensorflow port/implementation inspired by
# pyhank (https://github.com/etfrogers/pyhank Edward Rogers)


def iqdht(k_grid, fr, order=0):
    """Computes the inverse quasi-discrete Hankel transform for radial tensor, on the inner-most two dimensions

    Args:
        `k_grid` (tf.float): tensor corresponding to the angular frequency vector
        `fr` (`tf.float` or `tf.complex`): Field values on the radial grid of shape (..., 1, Nx).
        `order` (int, optional): Order of the inverse Hankel transform. Defaults to 0.

    Returns:
        `tf.float`: Radial grid corresponding to the iqdh-transformed data.
        `fr.dtype`: Inverse Hankel transform of the input data fr of shape (..., 1, Nx).
    """
    kr, ht = qdht(k_grid / 2 / np.pi, fr, order)

    return kr / 2 / np.pi, ht


def qdht(radial_grid, fr, order=0):
    """Implements a quasi-discrete Hankel transform for radial tensor data on the inner-most two dimensions of the input signal.

    Args:
        `radial_grid` (tf.float): 1D tensor of length N corresponding to the radial grid coordinates
        `fr` (tf.float or tf.complex): Real or complex field values on the radial grid of shape (.., 1, Nr).
        `order` (int, optional): Order of the Hankel transform. Defaults to 0.

    Returns:
        `tf.float`: tensor corresponding to the angular frequency vector
        `fr.dtype`: Hankel transform of the input data fr of shape (.., 1, Nr)
    """

    # Assert that the grid is a vector
    radial_grid = tf.squeeze(radial_grid)
    if radial_grid.ndim != 1:
        raise ValueError("QDHT: Radial grid should be a 1D vector")

    # If the tensor rank is not 3, then readjust for the calculation. In the end, we reshape back then return.
    input_rank = len(fr.shape)
    init_shape = fr.shape
    if input_rank == 1:
        fr = fr[tf.newaxis, tf.newaxis]
    elif input_rank == 2:
        fr = fr[tf.newaxis]
    elif input_rank > 3:
        fr = tf.reshape(fr, [-1, *init_shape[-2:]])

    sdtype = fr.dtype
    rdtype = radial_grid.dtype
    n_points = len(radial_grid)
    m = fr.shape[0]

    ### Create the transformation matrix
    # Calculate N+1 roots; must be calculated before max_radius can be derived from k_grid
    alpha = scipy_bessel.jn_zeros(order, n_points + 1)
    alpha = alpha[0:-1]
    alpha_n1 = alpha[-1]

    # Calculate coordinate vectors
    max_radius = np.max(radial_grid)
    r = tf.cast(alpha * max_radius / alpha_n1, dtype=rdtype)
    v = alpha / (2 * np.pi * max_radius)
    kr = tf.cast(2 * np.pi * v, dtype=rdtype)
    v_max = alpha_n1 / (2 * np.pi * max_radius)
    S = alpha_n1

    # Calculate hankel matrix and vectors
    jp = tf.cast(scipy_bessel.jv(order, tf.linalg.matmul(tf.expand_dims(alpha, -1), tf.expand_dims(alpha, 0)) / S), sdtype)
    jp1 = tf.cast(np.abs(scipy_bessel.jv(order + 1, alpha)), dtype=sdtype)
    T = 2 * jp / tf.linalg.matmul(tf.expand_dims(jp1, -1), tf.expand_dims(jp1, 0)) / S
    JR = jp1 / max_radius
    JV = jp1 / v_max

    # Define the tf while loop conditional and body
    use_interp = tf.cond(
        tf.math.logical_or(sdtype == tf.complex64, sdtype == tf.complex128),
        lambda: helper_spline_complex,
        lambda: helper_spline_real,
    )

    def hankel_loopCond(idx, fr_transformed):
        return tf.less(idx, m)

    def hankel_loopBody(idx, fr_transformed):
        f_transform = use_interp(tf.math.reduce_min(radial_grid), tf.math.reduce_max(radial_grid), r, fr[idx, 0, :])

        ht = tf.expand_dims(JV, -1) * tf.linalg.matmul(T, tf.expand_dims((f_transform / JR), -1))
        ht = tf.reshape(ht, [1, 1, n_points])

        fr_transformed = tf.concat([fr_transformed, ht], axis=0)
        idx += 1
        return [idx, fr_transformed]

    # perform transformation in while loop
    fr_transformed = tf.zeros([1, 1, n_points], dtype=sdtype)
    idx = tf.constant(0, dtype=tf.int32)
    loopdata = tf.while_loop(
        hankel_loopCond,
        hankel_loopBody,
        loop_vars=[idx, fr_transformed],
        swap_memory=True,
    )

    # reshape the hankel transformed signal back to the user input batch_size
    ht_signal = loopdata[1][1:]
    if input_rank != 3:
        ht_signal = tf.reshape(ht_signal, init_shape)

    return kr, ht_signal
