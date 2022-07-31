from dflat.physical_optical_layer.core.ms_parameterization import get_cartesian_grid
import tensorflow as tf
import numpy as np


def assemble_ER_superEllipse(rcwa_parameters, lay_eps, args):
    # args[0] = len_x
    # args[1] = len_y
    # args[2] = n (power)

    x_mesh, y_mesh = get_cartesian_grid(
        rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"]
    )

    # Generate SuperEllipse shape
    val = (np.abs(2 * x_mesh / args[0]) ** args[2] + np.abs(2 * y_mesh / args[1]) ** args[2]) - 1
    val[np.where(val >= 0)] = 0.0
    val[np.where(val < 0)] = 1.0
    val = val[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]

    ER_meta = lay_eps + (rcwa_parameters["erd"] - lay_eps) * val

    return ER_meta


def assemble_ER_rectangular_fin(rcwa_parameters, lay_eps, args):
    # args[0] = len_x
    # args[1] = len_y
    x_mesh, y_mesh = get_cartesian_grid(
        rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"]
    )

    ## Generate Rectangle fin shape
    val = np.zeros_like(x_mesh)
    val[(np.abs(2 * x_mesh / args[0]) <= 1.0) & (np.abs(2 * y_mesh / args[1]) <= 1.0)] = 1.0
    ER_meta = lay_eps + (rcwa_parameters["erd"] - lay_eps) * val

    return ER_meta
