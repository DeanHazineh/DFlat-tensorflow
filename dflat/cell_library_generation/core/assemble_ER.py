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


def assemble_double_nanofins(rcwa_parameters, lay_eps, args):
    # args[0] = len1_x
    # args[1] = len1_y
    # args[2] = len2_x
    # args[3] = len2_y
    # args[4] = offset

    y_mesh, x_mesh = get_cartesian_grid(
        rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"]
    )

    ## Generate Rectangle fin shape
    val = np.zeros_like(x_mesh)
    val[(np.abs(2 * (x_mesh - args[4]) / args[0]) <= 1.0) & (np.abs(2 * (y_mesh - args[5]) / args[1]) <= 1.0)] = 1.0
    val[(np.abs(2 * (x_mesh + args[4]) / args[2]) <= 1.0) & (np.abs(2 * (y_mesh + args[5]) / args[3]) <= 1.0)] = 1.0

    # import matplotlib.pyplot as plt

    # print(val.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(val)
    # plt.show()

    ER_meta = lay_eps + (rcwa_parameters["erd"] - lay_eps) * val

    return ER_meta
