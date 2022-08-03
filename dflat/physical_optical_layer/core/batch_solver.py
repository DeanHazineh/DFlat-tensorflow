import tensorflow as tf
from .ms_parameterization import generate_cell_perm
from .colburn_solve_field import simulate
import dflat.data_structure as datstruc

reset_keys = ["wavelength_set_m", "thetas", "phis", "pte", "ptm", "batch_wavelength_dim"]


def generate_simParam_set(rcwa_parameters):

    # rcwa_parameters["batch_wavelength_dim"] must be True to use this

    # Unpack the input parameters
    wavelength_set_m = rcwa_parameters["wavelength_set_m"]
    num_wavelengths = len(wavelength_set_m)
    thetas = rcwa_parameters["thetas"]
    phis = rcwa_parameters["phis"]
    pte = rcwa_parameters["pte"]
    ptm = rcwa_parameters["ptm"]

    rcwa_parameters_list = []
    for iter in range(num_wavelengths):
        rcwa_settings = rcwa_parameters.get_dict()
        for key in reset_keys:
            del rcwa_settings[key]

        rcwa_settings["wavelength_set_m"] = [wavelength_set_m[iter]]
        rcwa_settings["thetas"] = [thetas[iter]]
        rcwa_settings["phis"] = [phis[iter]]
        rcwa_settings["pte"] = [pte[iter]]
        rcwa_settings["ptm"] = [ptm[iter]]
        rcwa_settings["batch_wavelength_dim"] = False
        rcwa_parameters_list.append(datstruc.rcwa_params(rcwa_settings))

    return rcwa_parameters_list


def full_rcwa_shape(norm_param, rcwa_parameters):
    ### NOTE: Transmittance is returned here!!! Not Transmission.

    Er, Ur = generate_cell_perm(norm_param, rcwa_parameters)

    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    outputs = simulate(Er, Ur, rcwa_parameters)
    tx = outputs["tx"][:, :, :, PQ_zero, 0]
    ty = outputs["ty"][:, :, :, PQ_zero, 0]

    trans = tf.stack([tf.math.abs(tx), tf.math.abs(ty)])
    trans = tf.transpose(trans, perm=[1, 0, 3, 2])
    phase = tf.stack([tf.math.angle(tx), tf.math.angle(ty)])
    phase = tf.transpose(phase, perm=[1, 0, 3, 2])

    return (trans, phase)


def batched_wavelength_rcwa_shape(norm_param, rcwa_parameters):
    ### NOTE: Transmittance is returned here!!! Not Transmission.

    rcwa_parameters_list = generate_simParam_set(rcwa_parameters)
    num_wavelengths = len(rcwa_parameters_list)

    def lambda_loopCond(idx_, hold_trans_, hold_phase_):
        return tf.less(idx_, num_wavelengths)

    def lambda_loopBody(idx_, hold_trans_, hold_phase_):
        trans, phase = full_rcwa_shape(norm_param, rcwa_parameters_list[idx_])

        hold_trans_ = tf.concat([hold_trans_, trans], axis=0)
        hold_phase_ = tf.concat([hold_phase_, phase], axis=0)

        idx_ += 1

        return [idx_, hold_trans_, hold_phase_]

    # Run batched loop
    dtype = rcwa_parameters["dtype"]
    idx = tf.constant(0, dtype=tf.int32)
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    hold_trans = tf.zeros(shape=(1, 2, pixelsY, pixelsX), dtype=dtype)
    hold_phase = tf.zeros(shape=(1, 2, pixelsY, pixelsX), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, hold_trans, hold_phase])

    return (
        tf.stack(loopData[1][1:]),
        tf.stack(loopData[2][1:]),
    )
