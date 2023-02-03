import tensorflow as tf

import dflat.data_structure as datstruc
from .colburn_solve_field import simulate
from .ms_parameterization import generate_cell_perm
from copy import deepcopy


def generate_simParam_set(rcwa_parameters):
    # In most cases "batch_wavelength_dim" would be true but i want this to work when it is false also
    original_dict = rcwa_parameters.get_original_dict()
    num_wavelengths = len(original_dict["wavelength_set_m"])

    rcwa_parameters_list = []
    for iter in range(num_wavelengths):
        rcwa_dict = deepcopy(original_dict)
        rcwa_dict["wavelength_set_m"] = [original_dict["wavelength_set_m"][iter]]
        rcwa_dict["thetas"] = [original_dict["thetas"][iter]]
        rcwa_dict["phis"] = [original_dict["phis"][iter]]
        rcwa_dict["pte"] = [original_dict["pte"][iter]]
        rcwa_dict["ptm"] = [original_dict["ptm"][iter]]
        rcwa_dict["batch_wavelength_dim"] = False
        rcwa_parameters_list.append(datstruc.rcwa_params(rcwa_dict))

    return rcwa_parameters_list


def full_rcwa_shape(norm_param, rcwa_parameters, cell_parameterization, feature_layer):
    ### Returns the complex field

    Er, Ur = generate_cell_perm(norm_param, rcwa_parameters, cell_parameterization, feature_layer)

    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    outputs = simulate(Er, Ur, rcwa_parameters)
    tx = outputs["tx"][:, :, :, PQ_zero, 0]
    ty = outputs["ty"][:, :, :, PQ_zero, 0]

    return tf.transpose(tf.stack([tx, ty]), [1, 0, 3, 2])


def batched_wavelength_rcwa_shape(norm_param, rcwa_parameters, cell_parameterization, feature_layer):

    rcwa_parameters_list = generate_simParam_set(rcwa_parameters)
    num_wavelengths = len(rcwa_parameters_list)

    def lambda_loopCond(idx_, hold_field_):
        return tf.less(idx_, num_wavelengths)

    def lambda_loopBody(idx_, hold_field_):
        field = full_rcwa_shape(norm_param, rcwa_parameters_list[idx_], cell_parameterization, feature_layer)
        hold_field_ = tf.concat([hold_field_, field], axis=0)
        idx_ += 1
        return [idx_, hold_field_]

    # Run batched loop
    cdtype = rcwa_parameters["cdtype"]
    idx = tf.constant(0, dtype=tf.int32)
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    hold_field = tf.zeros(shape=(1, 2, pixelsY, pixelsX), dtype=cdtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, hold_field])

    return tf.stack(loopData[1][1:])


def compute_ref_field(rcwa_parameters):
    # Compute the reference field without batching
    if rcwa_parameters["batch_wavelength_dim"]:
        rcwa_settings = rcwa_parameters.get_original_dict()
        rcwa_settings["batch_wavelength_dim"] = False
        rcwa_parameters = datstruc.rcwa_params(rcwa_settings)

    # Retrieve simulation size parameters
    batchSize = rcwa_parameters["batchSize"]
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    Nlay = rcwa_parameters["Nlay"]
    Nx = rcwa_parameters["Nx"]
    Ny = rcwa_parameters["Ny"]
    cdtype = rcwa_parameters["cdtype"]
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    lay_eps_list = rcwa_parameters["lay_eps_list"]
    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2

    Ur = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    Er = tf.concat([lay_eps_list[i] * tf.ones(materials_shape_lay, dtype=cdtype) for i in range(Nlay)], axis=3)

    outputs = simulate(Er, Ur, rcwa_parameters)
    tx = outputs["tx"][:, :, :, PQ_zero, 0]
    ty = outputs["ty"][:, :, :, PQ_zero, 0]

    return tf.transpose(tf.stack([tx, ty]), [1, 0, 3, 2])
