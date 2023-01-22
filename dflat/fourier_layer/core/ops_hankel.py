# The QDHT code included here is a tensorflow port/implementation inspired by pyhank (https://github.com/etfrogers/pyhank Edward
# Rogers)


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.special as scipy_bessel


def radial_crop_or_pad(image, output_size):
    """Generalized tf.resize_with_crop_or_pad in a way that is valid for radial data vectors. This is because crop or
    pad runs an inner crop whereas we want to pad or crop only the outer/rhs of the tensor

    Args:
        `image` (tf.float): radial image to be resized via pad or crop, of shape (batch_size, 1, Nr).
        `output_size` (dict): target output dimensions of form {"x": int, "y": float, "r": float}
    """
    current_r = image.shape[2]
    target_r = output_size["r"]

    if target_r < current_r:
        cropby = current_r - target_r
        image = tf.pad(image, [[0, 0], [0, 0], [cropby, 0]], mode="CONSTANT", constant_values=0)
        image = tf.squeeze(tf.image.resize_with_crop_or_pad(tf.expand_dims(image, -1), 1, target_r), -1)
    elif target_r > current_r:
        padby = target_r - current_r
        image = tf.pad(image, [[0, 0], [0, 0], [0, padby]], mode="CONSTANT", constant_values=0)
    else:
        image = image

    return image


def radial_conditional_resize_with_crop_or_pad(image, radial_symmetry, output_size):
    """Properly wraps tf.image.resize_with_crop_or_pad to conditionally treat radial data vectors along with 2D

    Args:
        `image` (tf.float): Image to be processed, of shape (batch_size, calc_samplesN['y'], calc_samplesN['x'])
            or (batch_size, 1, calc_samplesN["r"]).
        `radial_symmetry` (bool): Radial symmetry flag denoting if image is a batch of 2D data or batch of radial vector data
        `output_size` (dict): target output dimensions of form {"x": int, "y": float, "r": float}

    Returns:
        `tf.float`: Cropped or padded image, of shape (batch_size, output_size["y"], output_size["x"])
            or (batch_size, 1, output_size["r"]).
    """
    # if radial_symmetry False, use resize_with_crop_or_pad directly without issues, otherwise branch custom function
    image = tf.cond(
        radial_symmetry,
        lambda: radial_crop_or_pad(image, output_size),
        lambda: tf.squeeze(tf.image.resize_with_crop_or_pad(tf.expand_dims(image, -1), output_size["y"], output_size["x"]), -1),
    )

    return image


def radial_2d_transform(r_array):
    """Transform a radial, real array (,N) to a 2D profile (,2N-1, 2N-1).

    Args:
        `r_array` (float): Input radial vector/tensor of shape (batch_shape, N).

    Returns:
        `tf.float`: 2D-converted data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """
    N = int(r_array.shape[-1])
    batch_shape = tf.shape(r_array)[:-1]
    x_r = tf.reshape(r_array, [-1, N])

    xx, yy = tf.meshgrid(tf.range(1 - N, N), tf.range(1 - N, N))
    r = tf.sqrt(tf.cast(xx**2 + yy**2, dtype=x_r.dtype))

    x_2d = tfp.math.interp_regular_1d_grid(r, 0, N - 1, x_r, fill_value_above=0, fill_value_below=0)
    x_2d = tf.reshape(x_2d, tf.concat([batch_shape, tf.shape(x_2d)[1::]], 0))
    return x_2d


def radial_2d_transform_wrapped_phase(r_array):
    """Transform a radial, real array of phase values in radians (,N) to a 2D phase profile (,2N-1, 2N-1).

    This function is analogous to radial_2d_transform but properly interpolates the phase-wrapping discontinuity.

    Args:
        `r_array` (float): Input radial vector/tensor of shape (batch_shape, N).

    Returns:
        `tf.float`: 2D-converted phase data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """

    realTrans = radial_2d_transform(tf.cos(r_array))
    imagTrans = radial_2d_transform(tf.sin(r_array))

    return tf.math.atan2(imagTrans, realTrans)


def radial_2d_transform_complex(r_array):
    """Transform a radial, complex array (,N) to a 2D profile (, 2N-1, 2N-1).

    This function is analogous to radial_2d_transform but handles complex data.

    Args:
        `r_array` (tf.complex128): Input radial tensor of shape (batch_shape, N).

    Returns:
        `tf.complex128`: 2D-converted data via tensor of shape (batch_shape, 2N-1, 2N-1).
    """
    if r_array.dtype == tf.complex64:
        usedtype = tf.float32
    else:
        usetype = tf.float64
    TF_ZERO = tf.cast(0.0, dtype=usetype)

    radial_trans = radial_2d_transform(tf.math.abs(r_array))
    radial_phase = radial_2d_transform_wrapped_phase(tf.math.angle(r_array))

    return tf.complex(radial_trans, TF_ZERO) * tf.exp(tf.complex(TF_ZERO, radial_phase))


def helper_spline_complex(r_ref_min, r_ref_max, r, fr):
    f_transform_abs = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.math.abs(fr))
    f_transform_real = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.cos(tf.math.angle(fr)))
    f_transform_imag = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.sin(tf.math.angle(fr)))

    if fr.dtype == tf.complex64:
        usetype = tf.float32
    else:
        usetype = tf.float64
    TF_ZERO = tf.cast(0.0, dtype=usetype)

    return tf.complex(f_transform_abs, TF_ZERO) * tf.exp(tf.complex(TF_ZERO, tf.math.atan2(f_transform_imag, f_transform_real)))


def helper_spline_real(r_ref_min, r_ref_max, r, fr):
    f_transform = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, fr)
    return f_transform


def tf_generalSpline_regular1DGrid(r_ref, r, fr):
    """Computes the 1D interpolation of a complex tensor on a regular grid.

    This is a helper function wrapping tfp.math.interp_regular_1d_grid calls enabling easy use for
    complex or real data.

    Args:
        `r_ref` (tf.float64): tensor specifying the reference grid coordinates
        `r` (tf.float64): r values of the interpolated grid
        `fr` (tf.complex128): reference data corresponding to r_ref to interpolate over, of size (..., Nx). Interpolation
            is taken over the inner-most dimension, similar to the scipy.interp1d function.

    Returns:
        `tf.complex128`: New complex output interpolated on the regular grid r
    """
    # Note: Using this implementation to snap back to a uniform grid after the qdht calls is is not correct because the qdht returns non-uniform grids
    # However, we will use this for now to reduce time and since we find the errors to have negligible effect thus far

    dtype = fr.dtype
    interpFr = tf.cond(
        tf.math.logical_or(dtype == tf.complex64, dtype == tf.complex128),
        lambda: helper_spline_complex(tf.math.reduce_min(r_ref), tf.math.reduce_max(r_ref), r, fr),
        lambda: helper_spline_real(tf.math.reduce_min(r_ref), tf.math.reduce_max(r_ref), r, fr),
    )

    return interpFr


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
    tf.debugging.assert_equal(
        tf.shape(radial_grid).shape,
        1,
        message="QDHT: grid vector r must be 1d vector/tensor",
        summarize="Dimension check on grid r",
    )

    # Check the rank of the input signal
    # If the tensor rank is not 3, then readjust for the calculation. In the end, we reshape back then return.
    input_rank = tf.rank(fr)
    init_shape = fr.shape
    if tf.math.equal(input_rank, tf.TensorShape(2)):
        fr = tf.expand_dims(fr, 0)

    if tf.math.greater(input_rank, tf.TensorShape(3)):
        fr = tf.reshape(fr, [-1, init_shape[-2], init_shape[-1]])

    sdtype = fr.dtype
    rdtype = radial_grid.dtype
    n_points = radial_grid.shape[0]
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
