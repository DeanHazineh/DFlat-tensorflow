import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import os

from dflat.physical_optical_layer.core.ms_parameterization import get_cartesian_grid
from dflat.physical_optical_layer.core.colburn_solve_field import simulate
import dflat.plot_utilities.graphFunc as gF


def run_zeroOrder_library_gen(rcwa_parameters, paramlist, cell_fun, showDebugPlot=False, savepath=None, checkpoint_num=500):
    # Enforce that the batch_wavelength dim is false since it is slow if done this way
    if rcwa_parameters["batch_wavelength_dim"] == True:
        raise ValueError("For library generation, dont batch wavelengths! Run in CPU instead of GPU of out of Memory")

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
    layer_dielectric = 1  # in old version, this was utilized differently
    paramlist = tf.convert_to_tensor(paramlist, dtype=dtype)

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    lay_eps_list = rcwa_parameters["lay_eps_list"]

    # Assume unity magnetic permeability
    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    ER_list = []
    for lay_eps in lay_eps_list:
        ER_list.append(lay_eps * tf.ones(materials_shape_lay, dtype=cdtype))

    ### Get a reference field (reference field is crucial to interpret outputs I find)
    outputs = simulate(tf.concat(values=ER_list, axis=3), UR, rcwa_parameters)
    tx_ref = outputs["tx"][:, 0, 0, PQ_zero, 0]
    ty_ref = outputs["ty"][:, 0, 0, PQ_zero, 0]
    ref_field = tf.expand_dims(tf.transpose(tf.stack((tx_ref, ty_ref))), 0)

    ### Assemble the cells structure and simulate each shape
    ## Load from previous savepath checkpoint if it exists
    if savepath and os.path.exists(savepath + "Checkpoint.pickle"):
        print("Resuming from previous checkpoint")
        with open(savepath + "Checkpoint.pickle", "rb") as handle:
            checkpoint = pickle.load(handle)
            hold_field_zero_order = tf.convert_to_tensor(checkpoint["hold_field_zero_order"], cdtype)
            i_start = checkpoint["i"]
            print(i_start)
    else:
        hold_field_zero_order = tf.zeros(shape=(1, batchSize, 2), dtype=cdtype)
        i_start = -1

    for i in np.arange(i_start + 1, paramlist.shape[0], 1):
        start = time.time()

        ## Generate shape
        ER_struct = cell_fun(rcwa_parameters, lay_eps_list[layer_dielectric - 1], paramlist[i, :])
        ER_list[layer_dielectric - 1] = ER_struct
        ER = tf.concat(values=ER_list, axis=3)

        ### Call Simulation
        outputs = simulate(ER, UR, rcwa_parameters)
        tx = outputs["tx"][:, 0, 0, PQ_zero, 0]
        ty = outputs["ty"][:, 0, 0, PQ_zero, 0]

        field = tf.expand_dims(tf.transpose(tf.stack((tx, ty))), 0)
        hold_field_zero_order = tf.concat((hold_field_zero_order, field), axis=0)

        end = time.time()
        print("Progress: ", f"{i / paramlist.shape[0] * 100:3.2f}", " Step: ", i, " Time Elapsed: ", end - start)

        # Display the cell
        if showDebugPlot:
            # Define constants used to format the plot
            x_mesh, y_mesh = get_cartesian_grid(Lx, Nx, Ly, Ny)
            xmin_nm = np.min(x_mesh) * 1e9
            ymin_nm = np.min(y_mesh) * 1e9
            xmax_nm = np.max(x_mesh) * 1e9
            ymax_nm = np.max(y_mesh) * 1e9
            erd_abs = np.abs(rcwa_parameters["erd"])

            if i == (i_start + 1):
                fig = plt.figure()
                ax = gF.addAxis(fig, 1, Nlay)
                images = []
                for j in range(Nlay):
                    image = ax[j].imshow(
                        np.abs(ER[0, 0, 0, j, :, :]),
                        extent=(xmin_nm, xmax_nm, ymin_nm, ymax_nm),
                        vmin=1,
                        vmax=erd_abs[0],
                    )
                    images.append(image)
            else:
                for j in range(Nlay):
                    images[j].set_data(np.abs(ER[0, 0, 0, j, :, :]))
            plt.pause(1e-1)

        # Save checkpoint in case of early termination
        # This is important for long runs where one might stop the code prematurely
        if savepath and (np.mod(i, checkpoint_num) == 0):
            print("saving checkpoint at step: ", i)
            data = {"hold_field_zero_order": hold_field_zero_order.numpy(), "i": i}
            with open(savepath + "Checkpoint.pickle", "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # If a checkpoint file was made and the full run is finished, delete it
    if savepath:
        if os.path.exists(savepath + "Checkpoint.pickle"):
            os.remove(savepath + "Checkpoint.pickle")

    transmission = tf.abs(hold_field_zero_order) ** 2 / tf.abs(ref_field) ** 2
    phase = tf.math.angle(ref_field) - tf.math.angle(hold_field_zero_order)
    plt.close()

    return transmission[1:].numpy(), phase[1:].numpy()
