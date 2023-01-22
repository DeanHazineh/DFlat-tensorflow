from dflat.physical_optical_layer.core.ms_parameterization import get_cartesian_grid
import tensorflow as tf
import numpy as np


def rotate_grid(xgrid, ygrid, theta):
    # Add the negative inside trig so that positive theta corresponds to clockwise rotations
    x_meshr = xgrid * tf.math.cos(-1 * theta) + ygrid * tf.math.sin(-1 * theta)
    y_meshr = -xgrid * tf.math.sin(-1 * theta) + ygrid * tf.math.cos(-1 * theta)
    return x_meshr, y_meshr


def assemble_ER_super_ellipse(rcwa_parameters, lay_eps, args, exp_power, inverse=False):
    # args[0] = len_x
    # args[1] = len_y
    # args[2] = rotation (optional)

    arg_len = len(args)
    sigmoid_coefficient = 1000.0
    if arg_len not in [2, 3]:
        raise ValueError("Rectangular fin assembly function expects shape parameters with 2 or 3 arguments")

    x_mesh, y_mesh = get_cartesian_grid(rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"])
    if arg_len == 3:  # If shape rotation is included, then rotate the coordinates
        x_mesh, y_mesh = rotate_grid(x_mesh, y_mesh, args[2])

    ## Generate super-ellipse binary (hole or pillar)
    fill = -1 if inverse else 1
    val = 1 - tf.math.abs(x_mesh * 2 / args[0]) ** exp_power - tf.math.abs(y_mesh * 2 / args[1]) ** exp_power
    val = 1 / (1 + np.exp(-1 * fill * sigmoid_coefficient * val))

    # Add dieletric to the layer of lay_eps embedding
    ER_meta = lay_eps + (rcwa_parameters["erd"] - lay_eps) * val

    return ER_meta


def assemble_ER_rectangular_resonator(rcwa_parameters, lay_eps, args):
    # exp_power = 20 is suitable for rectangle shape
    # args[0] = len_x
    # args[1] = len_y
    # args[2] = rotation (optional)

    return assemble_ER_super_ellipse(rcwa_parameters, lay_eps, args, exp_power=20, inverse=False)


def assemble_ER_inverse_rectangular_resonator(rcwa_parameters, lay_eps, args):
    # exp_power = 20 is suitable for rectangle shape
    # args[0] = len_x
    # args[1] = len_y
    # args[2] = rotation (optional)
    return assemble_ER_super_ellipse(rcwa_parameters, lay_eps, args, exp_power=20, inverse=True)


# def assemble_double_nanofins(rcwa_parameters, lay_eps, args):
#     # args[0] = len1_x
#     # args[1] = len1_y
#     # args[2] = len2_x
#     # args[3] = len2_y
#     # args[4] = offsetx
#     # args[5] = offsety
#     # args[6] = rotation theta (radians)
#     if len(args) != 7:
#         raise ValueError("Rectangular fin assembly function expects shape parameters with 7 arguments")

#     y_meshp, x_meshp = get_cartesian_grid(
#         rcwa_parameters["Lx"], rcwa_parameters["Nx"], rcwa_parameters["Ly"], rcwa_parameters["Ny"]
#     )
#     theta = args[6]
#     x_mesh = x_meshp * np.cos(theta) + y_meshp * np.sin(theta)
#     y_mesh = -x_meshp * np.sin(theta) + y_meshp * np.cos(theta)

#     val = np.zeros_like(x_mesh)
#     val[(np.abs(2 * (x_mesh - args[4]) / args[0]) <= 1.0) & (np.abs(2 * (y_mesh - args[5]) / args[1]) <= 1.0)] = 1.0
#     val[(np.abs(2 * (x_mesh + args[4]) / args[2]) <= 1.0) & (np.abs(2 * (y_mesh + args[5]) / args[3]) <= 1.0)] = 1.0

#     # import matplotlib.pyplot as plt

#     # print(val.shape)
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # ax.imshow(val)

#     return

#     # plt.show()

#     ER_meta = lay_eps + (rcwa_parameters["erd"] - lay_eps) * val

#     return ER_meta
