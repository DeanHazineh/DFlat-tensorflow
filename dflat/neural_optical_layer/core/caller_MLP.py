import tensorflow as tf
import numpy as np


def batched_broadband_MLP(norm_param, mlp_model, wavelength_m_asList, gridShape, output_stack_dim):
    """Returns the transmittance (Not transmission although the MLP was trained on transmission). Utilizes a for-loop
    to batch over input wavelenghts in order to not overload users memory

    Args:
        norm_param (tf.tensor): MLP input, normalized [0,1] of shape (N, D), where N is the number of cells and D is the shape parameter
        mlp_model (_type_): _description_
        wavelength_m_asList (_type_): _description_
        gridShape (_type_): _description_
        output_stack_dim (_type_): _description_

    Returns:
        _type_: _description_
    """

    ## Loop batch for wavelength
    num_wavelengths = len(wavelength_m_asList)
    dtype = mlp_model.get_model_dtype()

    def lambda_loopCond(idx_, hold_trans_, hold_phase_):
        return tf.less(idx_, num_wavelengths)

    def lambda_loopBody(idx_, hold_trans_, hold_phase_):
        # If the requested MLP takes wavelength_m as an input, then append wavelength to mlp input
        if mlp_model.get_wavelengthFlag():
            mlp_input = append_wavelength(norm_param, mlp_model.normalizeWavelength(wavelength_m_asList[idx_]), dtype)
        else:
            mlp_input = norm_param

        trans, phase = mlp_model.convert_output_complex(mlp_model(mlp_input), gridShape)
        # When we return sqrt(Trans), it is possible for some trans to be negative during model training which would
        # cause NaN values if this clipping is not done! It is important
        trans = tf.clip_by_value(trans, 0.0, 2.0)

        hold_trans_ = tf.concat([hold_trans_, tf.expand_dims(trans, 0)], axis=0)
        hold_phase_ = tf.concat([hold_phase_, tf.expand_dims(phase, 0)], axis=0)

        idx_ += 1
        return [idx_, hold_trans_, hold_phase_]

    # batch wavelength call
    idx = tf.constant(0, dtype=tf.int32)
    hold_trans = tf.zeros([1, output_stack_dim] + gridShape[1:], dtype=dtype)
    hold_phase = tf.zeros([1, output_stack_dim] + gridShape[1:], dtype=dtype)
    loopData = tf.while_loop(
        lambda_loopCond,
        lambda_loopBody,
        loop_vars=[idx, hold_trans, hold_phase],
        shape_invariants=[
            idx.get_shape(),
            tf.TensorShape([None, output_stack_dim] + gridShape[1:]),
            tf.TensorShape([None, output_stack_dim] + gridShape[1:]),
        ],
    )

    return (
        tf.math.sqrt(tf.stack(loopData[1][1:])),
        tf.stack(loopData[2][1:]),
    )


def flatten_reshape_shape_parameters(shape_vector):
    """Takes a shape/param vector of (D, PixelsY, PixelsX) and flattens to shape (PixelsY*PixelsX,D)"""
    vec_shape = tf.shape(shape_vector)
    return tf.reshape(tf.transpose(shape_vector, [1, 2, 0]), [vec_shape[1] * vec_shape[2], -1])


def normalize_params_np(unnorm_param, MLP_model):
    """(Helper function for neural optical models) Given an unormalized shape parameter vector and an MLP model,
    return the normalized params, [0,1].

    Args:
        `unnorm_param` (np.float): Unnormalized shape parameters for a cell, of form (N, D) where D is the shape degree.
        `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

    Returns:
        `np.float`: Model normalized parameter vector suitable for mlp input
    """
    paramList = [unnorm_param[:, i : i + 1] for i in range(unnorm_param.shape[1])]
    return np.hstack(MLP_model.normalizeInput(paramList))


def unnormalize_shapeVector_np(norm_param, MLP_model):
    """(Helper function for neural optical models) Given a model normalized shape parameter array, return the unnormalized
    length dimensions.

    Args:
        `norm_param` (np.float): Normalized shape parameters for a cell, of form (N, D) where D is the shape degree.
        `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

    Returns:
        `np.float`: The unnormalized parameter vector, where lengths are back in meaningful units of m.
    """
    databounds = MLP_model.get_preprocessDataBounds()
    shapeDegree = len(databounds) - 1

    return np.stack(
        [norm_param[:, i] * (databounds[i][1] - databounds[i][0]) + databounds[i][0] for i in range(shapeDegree)], -1
    )


def append_wavelength(p_param, normalized_wavelength_m, dtype):
    # This function takes a tensor shapevector of size (N,m) and appends the wavelength parameter
    # output is a modified tensor shapevector of size (N,m+1)

    # rather than normalizing wavelength here, we pulled out the function and mlp input and
    # require normalized wavelength to be passed in instead
    wavelength_mlp = tf.ones((p_param.shape[0], 1), dtype=dtype) * normalized_wavelength_m
    return tf.concat((p_param, wavelength_mlp), axis=-1)


def init_norm_param(init_type, dtype, gridShape, mlp_input_shape, init_args=[]):

    if init_type == "uniform":
        norm_param = 0.5 * tf.ones(shape=(mlp_input_shape - 1, gridShape[1], gridShape[2]), dtype=dtype)

    elif init_type == "random":
        norm_param = tf.random.uniform(shape=(mlp_input_shape - 1, gridShape[1], gridShape[2]), dtype=dtype)

    else:
        raise ValueError("initialize_norm_param: invalid init_type string;")

    return norm_param
