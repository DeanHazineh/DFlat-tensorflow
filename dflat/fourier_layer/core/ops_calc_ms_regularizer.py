import tensorflow as tf


def condResizeFn_true(ms_modulation_trans, ms_modulation_phase, parameters):
    method = "nearest"  # Do NOT change; This is a meaningful choice over interp methods!
    # Note that antialias has no effect when doing nearest neighbor upsampling

    # Expand the channel dimension of input matrices
    ms_modulation_trans = tf.expand_dims(ms_modulation_trans, -1)
    ms_modulation_phase = tf.expand_dims(ms_modulation_phase, -1)

    # handle radial flag conditional
    # tf v2 supports nested conditionals
    calc_samplesM = parameters["calc_samplesM"]
    radial_flag = parameters["radial_symmetry"]
    resizeTo = tf.cond(
        radial_flag,
        lambda: [1, calc_samplesM["r"]],
        lambda: [calc_samplesM["y"], calc_samplesM["x"]],
    )

    # Resize transmittance of the field -- Just upsampling so nearest interp req.
    calc_modulation_trans = tf.image.resize(
        ms_modulation_trans,
        resizeTo,
        method=method,
        preserve_aspect_ratio=False,
        antialias=False,
    )

    # Resize phase of the field -- Just upsampling so nearest interp req.
    calc_modulation_phase_real = tf.image.resize(
        tf.cos(ms_modulation_phase),
        resizeTo,
        method=method,
        preserve_aspect_ratio=False,
        antialias=False,
    )
    calc_modulation_phase_imag = tf.image.resize(
        tf.sin(ms_modulation_phase),
        resizeTo,
        method=method,
        preserve_aspect_ratio=False,
        antialias=False,
    )
    calc_modulation_phase = tf.atan2(calc_modulation_phase_imag, calc_modulation_phase_real)

    return tf.squeeze(calc_modulation_trans, -1), tf.squeeze(calc_modulation_phase, -1)


def condPad_true(calc_modulation_trans, calc_modulation_phase, parameters):
    # Get paddings from parameters
    padms_half = parameters["padms_half"]
    padhalfx = padms_half["x"]
    padhalfy = padms_half["y"]

    # handle radial flag conditional
    radial_flag = parameters["radial_symmetry"]
    paddings = tf.cond(
        radial_flag,
        lambda: [[0, 0], [0, 0], [0, padhalfx]],
        lambda: [[0, 0], [padhalfy, padhalfy], [padhalfx, padhalfx]],
    )

    calc_modulation_trans = tf.pad(calc_modulation_trans, paddings, mode="CONSTANT", constant_values=0)
    calc_modulation_phase = tf.pad(calc_modulation_phase, paddings, mode="CONSTANT", constant_values=0)

    return calc_modulation_trans, calc_modulation_phase


def regularize_ms_calc_tf(
    ms_modulation_trans,
    ms_modulation_phase,
    parameters,
):
    """Given an input amplitude and phase profile defined on the grid specified in the parameters object, upsample the
    field and pad according to the computed dimensions in prop_params object.

    Args:
        `ms_modulation_trans` (tf.float64): Metasurface transmittance on the user specified grid of shape
            (..., ms_samplesM['y'], ms_samplesM['x']) or (..., 1, ms_samplesM['r']).
        `ms_modulation_phase` (tf.float64): Metasurface phase on the user specified grid of shape
            (..., ms_samplesM['y'], ms_samplesM['x']) or (..., 1, ms_samplesM['r']).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Upsampled and padded metasurface transmittance of shape (..., calc_samplesN['y'], calc_samplesN['x'])
            or (..., 1, calc_samplesN['r']).
        `tf.float64`: Upsampled and padded metasurface phase of shape (..., calc_samplesN['y'], calc_samplesN['x'])
            or (..., 1, calc_samplesN['r'])
    """
    if not (ms_modulation_trans.shape == ms_modulation_phase.shape):
        raise ValueError("transmittance and phase must be the same shape")

    ### Handle multi-dimension input for different downstream use cases
    input_rank = len(ms_modulation_phase.shape)
    init_shape = ms_modulation_phase.shape
    if input_rank == 2:
        ms_modulation_phase = tf.expand_dims(ms_modulation_phase, 0)
        ms_modulation_trans = tf.expand_dims(ms_modulation_trans, 0)
    elif input_rank > 3:
        ms_modulation_phase = tf.reshape(ms_modulation_phase, [-1, init_shape[-2], init_shape[-1]])
        ms_modulation_trans = tf.reshape(ms_modulation_trans, [-1, init_shape[-2], init_shape[-1]])

    ### unpack parameters and handle radial flag appropriately
    dtype = parameters["dtype"]
    calc_samplesM = parameters["calc_samplesM"]
    ms_samplesM = parameters["ms_samplesM"]

    ### Resample the metasurface via nearest neighbors if required
    if (calc_samplesM["x"] > ms_samplesM["x"]) or (calc_samplesM["y"] > ms_samplesM["y"]):
        calc_modulation_trans, calc_modulation_phase = condResizeFn_true(ms_modulation_trans, ms_modulation_phase, parameters)
    else:
        calc_modulation_trans, calc_modulation_phase = (
            ms_modulation_trans,
            ms_modulation_phase,
        )

    ### Pad the array if samplesN (padded) is larger than samplesM (unpadded)
    calc_samplesN = parameters["calc_samplesN"]
    if (calc_samplesN["x"] > calc_samplesM["x"]) or (calc_samplesN["y"] > calc_samplesM["y"]):
        calc_modulation_trans, calc_modulation_phase = condPad_true(calc_modulation_trans, calc_modulation_phase, parameters)

    # Return with the same batch_size shape
    new_shape = calc_modulation_trans.shape
    if input_rank != 3:
        calc_modulation_trans = tf.reshape(calc_modulation_trans, [*init_shape[:-2], *new_shape[-2:]])
        calc_modulation_phase = tf.reshape(calc_modulation_phase, [*init_shape[:-2], *new_shape[-2:]])

    return tf.cast(calc_modulation_trans, dtype), tf.cast(calc_modulation_phase, dtype)
