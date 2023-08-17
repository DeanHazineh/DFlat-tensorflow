import tensorflow as tf


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

    ## Unpack Parameters
    out_dtype = mlp_model.get_model_dtype()
    num_wavelengths = len(wavelength_m_asList)
    output_stack_dim = mlp_model.get_output_pol_state()
    wavelength_input_flag = mlp_model.get_wavelengthFlag()
    if wavelength_input_flag:
        tf_ones = tf.ones([norm_param.shape[0], 1], dtype=tf.float32)

    def lambda_loopCond(idx_, hold_trans_, hold_phase_):
        return tf.less(idx_, num_wavelengths)

    def lambda_loopBody(idx_, hold_trans_, hold_phase_):
        # If the requested MLP takes wavelength_m as an input, then append wavelength to mlp input
        if mlp_model.get_wavelengthFlag():
            norm_wl = mlp_model.normalizeWavelength(wavelength_m_asList[idx_])
            mlp_input = tf.concat((norm_param, norm_wl * tf_ones), axis=-1)
        else:
            mlp_input = norm_param

        trans, phase = mlp_model.convert_output_complex(mlp_model(mlp_input), gridShape)
        hold_trans_ = tf.concat([hold_trans_, tf.expand_dims(trans, 0)], axis=0)
        hold_phase_ = tf.concat([hold_phase_, tf.expand_dims(phase, 0)], axis=0)

        idx_ += 1
        return [idx_, hold_trans_, hold_phase_]

    # batch wavelength call
    idx = tf.constant(0, dtype=tf.int32)
    hold_trans = tf.zeros([1, output_stack_dim] + gridShape[1:], dtype=out_dtype)
    hold_phase = tf.zeros([1, output_stack_dim] + gridShape[1:], dtype=out_dtype)
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

    return tf.stack(loopData[1][1:]), tf.stack(loopData[2][1:])
