from dflat.physical_optical_layer.core.ms_parameterization import get_cartesian_grid
import tensorflow as tf
import numpy as np


def tf_rotate_coord(xcart, ycart, theta):
    xr = xcart * tf.math.cos(theta) + ycart * tf.math.sin(theta)
    yr = -xcart * tf.math.sin(theta) + ycart * tf.math.cos(theta)
    return xr, yr


def assemble_ER_super_ellipse(rcwa_parameters, lay_eps, params, args):
    # params[0] = len_x
    # params[1] = len_y
    # params[2] = rotation (optional)
    # args[0] = exp_power
    # args[1] = inverse True vs False

    exp_power = args[0]
    inverse = args[1]
    sigmoid_coefficient = 100.0

    param_len = len(params)
    if param_len not in [2, 3]:
        raise ValueError("Rectangular fin assembly function expects shape parameters with 2 or 3 arguments")

    x_mesh, y_mesh = get_cartesian_grid(rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"])
    if param_len == 3:  # If shape rotation is included, then rotate the coordinates
        x_mesh, y_mesh = tf_rotate_coord(x_mesh, y_mesh, params[2])

    ## Generate super-ellipse binary (hole or pillar)
    fill = -1 if inverse else 1
    val = 1 - tf.math.abs(x_mesh * 2 / params[0]) ** exp_power - tf.math.abs(y_mesh * 2 / params[1]) ** exp_power
    val = 1 / (1 + np.exp(-1 * fill * sigmoid_coefficient * val))

    # Add dieletric to the layer of lay_eps embedding
    return lay_eps + (rcwa_parameters["erd"] - lay_eps) * val


def assemble_ER_rectangular_resonator(rcwa_parameters, lay_eps, params):
    # Note that exp_power = 20 is suitable for rectangle shape
    # params[0] = len_x
    # params[1] = len_y
    # params[2] = rotation (optional)

    return assemble_ER_super_ellipse(rcwa_parameters, lay_eps, params, [20.0, False])


def assemble_ER_inverse_rectangular_resonator(rcwa_parameters, lay_eps, params):
    # exp_power = 20 is suitable for rectangle shape
    # args[0] = len_x
    # args[1] = len_y
    # args[2] = rotation (optional)
    return assemble_ER_super_ellipse(rcwa_parameters, lay_eps, params, [20.0, True])
