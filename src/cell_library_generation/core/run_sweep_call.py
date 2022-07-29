import numpy as np
import tensorflow as tf
from physical_optical_layer.core.ms_parameterization import get_cartesian_grid
from physical_optical_layer.core.colburn_solve_field import simulate
import tools.graphFunc as gF
import matplotlib.pyplot as plt
import time


def run_zeroOrder_library_gen(rcwa_parameters, paramlist, cell_fun, showDebugPlot=False):
    ### Unpack some RCWA settings
    batchSize = rcwa_parameters["batchSize"]
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    Nx = rcwa_parameters["Nx"]
    Ny = rcwa_parameters["Ny"]
    Lx = rcwa_parameters["Lx"]
    Ly = rcwa_parameters["Ly"]
    Nlay = rcwa_parameters["Nlay"]
    dtype = rcwa_parameters["dtype"]
    cdtype = rcwa_parameters["cdtype"]
    TF_ZERO = tf.constant(0.0, dtype=dtype)
    TF_ONE = tf.constant(1.0, dtype=dtype)
    x_mesh, y_mesh = get_cartesian_grid(Lx, Nx, Ly, Ny)
    xmin_nm = np.min(x_mesh) * 1e9
    ymin_nm = np.min(y_mesh) * 1e9
    xmax_nm = np.max(x_mesh) * 1e9
    ymax_nm = np.max(y_mesh) * 1e9
    erd_abs = np.abs(rcwa_parameters["erd"])

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    lay_eps_list = rcwa_parameters["lay_eps_list"]

    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    ER_list = []
    for lay_eps in lay_eps_list:
        ER_list.append(lay_eps * tf.ones(materials_shape_lay, dtype=cdtype))

    ### Get a reference field
    outputs = simulate(tf.concat(values=ER_list, axis=3), UR, rcwa_parameters)
    tx_ref = outputs["tx"][:, 0, 0, PQ_zero, 0]
    ty_ref = outputs["ty"][:, 0, 0, PQ_zero, 0]
    ref_field = np.expand_dims(np.transpose(np.stack((tx_ref, ty_ref))), 0)

    ### Assemble the cells structure and simulate each shape
    hold_field_zero_order = np.zeros(shape=(paramlist.shape[0], batchSize, 2), dtype=np.complex64)
    for i in range(paramlist.shape[0]):
        start = time.time()

        ## Generate shape
        ER_struct = cell_fun(rcwa_parameters, lay_eps_list[1], paramlist[i, :])
        ER = tf.concat(values=[ER_list[0], ER_struct, ER_list[1]], axis=3)

        # Display the cell
        if showDebugPlot:
            if i == 0:
                fig = plt.figure()
                ax = gF.addAxis(fig, 1, 3)
                image1 = ax[0].imshow(
                    np.abs(ER[0, 0, 0, 0, :, :]), extent=(xmin_nm, xmax_nm, ymin_nm, ymax_nm), vmin=1, vmax=erd_abs[0]
                )
                image2 = ax[1].imshow(
                    np.abs(ER[0, 0, 0, 1, :, :]), extent=(xmin_nm, xmax_nm, ymin_nm, ymax_nm), vmin=1, vmax=erd_abs[0]
                )
                image3 = ax[2].imshow(
                    np.abs(ER[0, 0, 0, 2, :, :]), extent=(xmin_nm, xmax_nm, ymin_nm, ymax_nm), vmin=1, vmax=erd_abs[0]
                )
            else:
                image1.set_data(np.abs(ER[0, 0, 0, 0, :, :]))
                image2.set_data(np.abs(ER[0, 0, 0, 1, :, :]))
                image3.set_data(np.abs(ER[0, 0, 0, 2, :, :]))
            plt.pause(1e-3)

        ### Call Simulation
        outputs = simulate(ER, UR, rcwa_parameters)
        tx = outputs["tx"][:, 0, 0, PQ_zero, 0]
        ty = outputs["ty"][:, 0, 0, PQ_zero, 0]
        hold_field_zero_order[i, :, 0] = tx
        hold_field_zero_order[i, :, 1] = ty

        end = time.time()
        print("Progress: ", f"{i / paramlist.shape[0] * 100:3.2f}", " Step: ", i, " Time Elapsed: ", end - start)

    return ref_field, hold_field_zero_order
