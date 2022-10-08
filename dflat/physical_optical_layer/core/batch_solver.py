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
    ### Returns the complex field

    Er, Ur = generate_cell_perm(norm_param, rcwa_parameters)

    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    outputs = simulate(Er, Ur, rcwa_parameters)
    tx = outputs["tx"][:, :, :, PQ_zero, 0]
    ty = outputs["ty"][:, :, :, PQ_zero, 0]

    return tf.transpose(tf.stack([tx, ty]), [1, 0, 3, 2])


# Deprecated since it is so inneficient
def batched_wavelength_rcwa_shape(norm_param, rcwa_parameters):

    rcwa_parameters_list = generate_simParam_set(rcwa_parameters)
    num_wavelengths = len(rcwa_parameters_list)

    def lambda_loopCond(idx_, hold_field_):
        return tf.less(idx_, num_wavelengths)

    def lambda_loopBody(idx_, hold_field_):
        field = full_rcwa_shape(norm_param, rcwa_parameters_list[idx_])
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
        rcwa_settings = rcwa_parameters.get_dict()
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

    Ur = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    Er = []
    for i in range(Nlay):
        Er.append(lay_eps_list[i] * tf.ones(materials_shape_lay, dtype=cdtype))
    Er = tf.concat(Er, axis=3)

    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    outputs = simulate(Er, Ur, rcwa_parameters)

    tx = outputs["tx"][:, :, :, PQ_zero, 0]
    ty = outputs["ty"][:, :, :, PQ_zero, 0]

    return tf.transpose(tf.stack([tx, ty]), [1, 0, 3, 2])
