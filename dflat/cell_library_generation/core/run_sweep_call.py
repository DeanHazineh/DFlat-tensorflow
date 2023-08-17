import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
import os

from dflat.physical_optical_layer.core.ms_parameterization import get_cartesian_grid
from dflat.physical_optical_layer.core.colburn_solve_field import simulate
import dflat.plot_utilities.graphFunc as gF


def run_library_gen(rcwa_parameters, paramlist, cell_fun, fun_args=None, showDebugPlot=True, savepath=None, checkpoint_num=500, zero_order_only=True):
    # Enforce that the batch_wavelength dim is false since it is slow if done this way
    if rcwa_parameters["batch_wavelength_dim"] == True:
        raise ValueError("For library generation, dont batch wavelengths! Run in CPU instead of GPU of out of Memory")

    # Unpack some RCWA settings
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
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    lay_eps_list = rcwa_parameters["lay_eps_list"]

    # Assume unity magnetic permeability
    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    ER_list = [lay_eps * tf.ones(materials_shape_lay, dtype=cdtype) for lay_eps in lay_eps_list]

    # Get a reference field (reference field is crucial to interpret outputs I find)
    outputs = simulate(tf.concat(values=ER_list, axis=3), UR, rcwa_parameters)
    tx_ref = outputs["tx"][:, 0, 0, :, 0]
    ty_ref = outputs["ty"][:, 0, 0, :, 0]
    ref_field = tf.expand_dims(tf.transpose(tf.stack((tx_ref, ty_ref))), 0).numpy()
    if zero_order_only:
        ref_field = ref_field[:, PQ_zero, :, :]

    # Load from previous savepath checkpoint if it exists
    if savepath and os.path.exists(savepath + "Checkpoint.pickle"):
        print("Resuming from previous checkpoint")
        with open(savepath + "Checkpoint.pickle", "rb") as handle:
            checkpoint = pickle.load(handle)
            hold_field = checkpoint["hold_field"]
            i_start = checkpoint["i"]
    else:
        if zero_order_only:
            hold_field = np.zeros(shape=(paramlist.shape[0], batchSize, 2), dtype=complex)
        else:
            hold_field = np.zeros(shape=(paramlist.shape[0], np.prod(rcwa_parameters["PQ"]), batchSize, 2), dtype=complex)
        i_start = -1

    # Run library sweep
    for i in np.arange(i_start + 1, paramlist.shape[0], 1):
        start = time.time()
        ## Generate shape
        cell_params = tf.convert_to_tensor(paramlist[i, :], dtype)
        ER_struct = cell_fun(rcwa_parameters, lay_eps_list[layer_dielectric - 1], cell_params, fun_args)
        ER_list[layer_dielectric - 1] = ER_struct
        ER = tf.concat(values=ER_list, axis=3)

        ### Call Simulation
        outputs = simulate(ER, UR, rcwa_parameters)
        tx = outputs["tx"][:, 0, 0, :, 0]
        ty = outputs["ty"][:, 0, 0, :, 0]
        field = tf.expand_dims(tf.transpose(tf.stack((tx, ty))), 0)
        hold_field[i] = field.numpy()[:, PQ_zero, :, :] if zero_order_only else field.numpy()

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
        # This is helpful for long runs where one might stop the code prematurely
        if savepath and (np.mod(i, checkpoint_num) == 0):
            print("saving checkpoint at step: ", i)
            data = {"hold_field": hold_field, "i": i, "ref_field": ref_field}
            with open(savepath + "Checkpoint.pickle", "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save the simulation results
    transmission = np.abs(hold_field) ** 2 / (np.abs(ref_field) ** 2 + 1e-6)
    phase = np.angle(ref_field) - np.angle(hold_field)
    if savepath:
        data = {"hold_field": hold_field, "ref_field": ref_field, "transmission": transmission, "phase": phase, "paramlist": paramlist}
        with open(savepath + "Library_gen_output.pickle", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # If a checkpoint file was made and the full run is finished, delete it
    if savepath:
        if os.path.exists(savepath + "Checkpoint.pickle"):
            os.remove(savepath + "Checkpoint.pickle")
    plt.close()

    return transmission, phase
