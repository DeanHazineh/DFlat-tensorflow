import tensorflow as tf
import numpy as np


def batched_broadband_MLP(norm_param, mlp_model, wavelength_m_asList, gridShape):
    """Returns the transmittance (Not transmission although the MLP was trained on transmission). Utilizes a for-loop
    to batch over input wavelengths in order to not overload users memory

    Args:
        norm_param (tf.tensor): MLP input, normalized [0,1] of shape (N, D), where N is the number of cells and D is the shape parameter
        mlp_model (_type_): _description_
        wavelength_m_asList (_type_): _description_
        gridShape (_type_): _description_
    Returns:
        _type_: _description_
    """

    ## Loop batch for wavelength
    num_wavelengths = len(wavelength_m_asList)
    dtype = mlp_model.get_model_dtype()
    output_stack_dim = mlp_model.get_output_pol_state()

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


def append_wavelength(p_param, normalized_wavelength_m, dtype):
    # This function takes a tensor shapevector of size (N,m) and appends the wavelength parameter
    # output is a modified tensor shapevector of size (N,m+1)

    # rather than normalizing wavelength here, we pulled out the function and mlp input and
    # require normalized wavelength to be passed in instead
    wavelength_mlp = tf.ones((p_param.shape[0], 1), dtype=dtype) * normalized_wavelength_m
    return tf.concat((p_param, wavelength_mlp), axis=-1)
