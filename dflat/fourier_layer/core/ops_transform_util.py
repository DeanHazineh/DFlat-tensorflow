import tensorflow as tf
import tensorflow_probability as tfp


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
        image = image[..., :target_r]
    elif target_r > current_r:
        paddings = [[0, 0], [0, 0], [0, target_r - current_r]]
        image = tf.pad(image, paddings, mode="CONSTANT", constant_values=0)

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
    if radial_symmetry:
        image = radial_crop_or_pad(image, output_size)
    else:
        image = tf.squeeze(tf.image.resize_with_crop_or_pad(tf.expand_dims(image, -1), output_size["y"], output_size["x"]), -1)

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
    radial_trans = radial_2d_transform(tf.math.abs(r_array))
    radial_phase = radial_2d_transform_wrapped_phase(tf.math.angle(r_array))
    TF_ZERO = tf.cast(0.0, dtype=radial_trans.dtype)

    return tf.complex(radial_trans, TF_ZERO) * tf.exp(tf.complex(TF_ZERO, radial_phase))


def helper_spline_complex(r_ref_min, r_ref_max, r, fr):
    f_transform_abs = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.math.abs(fr))
    f_transform_real = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.cos(tf.math.angle(fr)))
    f_transform_imag = tfp.math.interp_regular_1d_grid(r, r_ref_min, r_ref_max, tf.sin(tf.math.angle(fr)))

    TF_ZERO = tf.cast(0.0, dtype=f_transform_abs.dtype)
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

    if not tf.is_tensor(fr):
        fr = tf.convert_to_tensor(fr)

    dtype = fr.dtype
    if dtype == tf.complex64 or dtype == tf.complex128:
        interpFr = helper_spline_complex(tf.math.reduce_min(r_ref), tf.math.reduce_max(r_ref), r, fr)
    else:
        interpFr = helper_spline_real(tf.math.reduce_min(r_ref), tf.math.reduce_max(r_ref), r, fr)

    return interpFr
