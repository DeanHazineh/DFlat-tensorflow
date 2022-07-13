import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

sys.path.append(".")

from data_structure import rcwa_params as rcwa_params
from physical_optical_layer.core.ms_parameterization import get_cartesian_grid
from physical_optical_layer.core.colburn_solve_field import simulate
import tools.graphFunc as gF
import matplotlib.pyplot as plt


def sweepLibrary_Nanofin(rcwa_parameters, paramlist, showDebugPlot=False):

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

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    PQ_zero = tf.math.reduce_prod(rcwa_parameters["PQ"]) // 2
    lay_eps_list = rcwa_parameters["lay_eps_list"]

    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)
    ER_dev0 = lay_eps_list[0] * tf.ones(materials_shape_lay, dtype=cdtype)
    ER_dev1 = lay_eps_list[1] * tf.ones(materials_shape_lay, dtype=cdtype)
    ER_dev2 = lay_eps_list[2] * tf.ones(materials_shape_lay, dtype=cdtype)

    x_mesh, y_mesh = get_cartesian_grid(Lx, Nx, Ly, Ny)

    ### Get a reference field
    ER = tf.concat(values=[ER_dev0, ER_dev1, ER_dev2], axis=3)
    outputs = simulate(ER, UR, rcwa_parameters)
    tx_ref = outputs["tx"][:, 0, 0, PQ_zero, 0]
    ty_ref = outputs["ty"][:, 0, 0, PQ_zero, 0]
    ref_field = np.transpose(np.stack((tx_ref, ty_ref)))

    ### Assemble the cells structure and simulate each shape
    hold_field_zero_order = np.zeros(shape=(paramlist.shape[0], batchSize, 2), dtype=np.complex64)
    for i in range(paramlist.shape[0]):
        start = time.time()

        ## Generate Rectangle fin shape
        val = np.zeros_like(x_mesh)
        val[(np.abs(x_mesh / (paramlist[i, 0] / 2)) <= 1.0) & (np.abs(y_mesh / (paramlist[i, 1] / 2)) <= 1.0)] = 1.0
        ER_meta = lay_eps_list[1] + (rcwa_parameters["erd"] - lay_eps_list[1]) * val
        ER = tf.concat(values=[ER_dev0, ER_meta, ER_dev2], axis=3)

        # Display the cell
        if showDebugPlot:
            if i == 0:
                fig = plt.figure()
                ax = gF.addAxis(fig, 1, 3)
                image1 = ax[0].imshow(np.abs(ER[0, 0, 0, 0, :, :]))
                image2 = ax[1].imshow(np.abs(ER[0, 0, 0, 1, :, :]))
                image3 = ax[2].imshow(np.abs(ER[0, 0, 0, 2, :, :]))
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

        # ## Every so often, save a checkpoint just in case:
        # if np.mod(i, 1000) == 0:
        #     timestr = time.strftime("%Y%m%d-%H%M%S")
        #     data = {"hold_field_zero_order": hold_field_zero_order}
        #     with open(timestr + ".pickle", "wb") as handle:
        #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ref_field, hold_field_zero_order


def call_library_generation_nanoFin(savepath, FM):

    ### Specify RCWA Solver parameters
    wavelength_set_m = [400e-9, 450e-9, 500e-9, 550e-9]
    rcwa_settings = {
        "wavelength_set_m": wavelength_set_m,
        "thetas": [0.0 for i in wavelength_set_m],
        "phis": [0.0 for i in wavelength_set_m],
        "pte": [1.0 for i in wavelength_set_m],
        "ptm": [1.0 for i in wavelength_set_m],
        "pixelsX": 1,
        "pixelsY": 1,
        "PQ": [FM, FM],
        "Lx": 350e-9,
        "Ly": 350e-9,
        "L": [1e-3, 600.0e-9, 1e-3],
        "Lay_mat": ["SiO2", "Vacuum", "Vacuum"],
        "material_dielectric": "TiO2",
        "er1": "SiO2",
        "er2": "Vacuum",
        "Nx": 512,
        "Ny": 512,
        "parameterization_type": "None",
        "batch_wavelength_dim": False,
        "dtype": tf.float32,
        "cdtype": tf.complex64,
    }
    rcwa_parameters = rcwa_params(rcwa_settings, bare=True)

    ### Define sweep ranges
    len_x = np.arange(60e-9, 300e-9, 5e-9)
    len_y = np.arange(60e-9, 300e-9, 5e-9)
    Len_x, Len_y = np.meshgrid(len_x, len_y)
    paramlist = np.transpose(np.vstack((Len_x.flatten(), Len_y.flatten())))

    ### Run SuperEllipse Sweep
    ref_field, hold_field_zero_order = sweepLibrary_Nanofin(rcwa_parameters, paramlist, showDebugPlot=False)

    trans = np.abs(hold_field_zero_order) ** 2
    ref_trans = np.abs(np.expand_dims(ref_field, 0)) ** 2
    phase = np.angle(hold_field_zero_order)
    ref_phase = np.angle(np.expand_dims(ref_field, 0))

    trans = trans / ref_trans
    phase = phase - ref_phase
    trans = trans.reshape([Len_x.shape[0], Len_x.shape[1], len(wavelength_set_m), 2])
    phase = phase.reshape([Len_x.shape[0], Len_x.shape[1], len(wavelength_set_m), 2])

    ### Save the data
    data = {
        "trans": trans,
        "phase": phase,
        "len_x": len_x,
        "len_y": len_y,
        "wavelength_set_m": wavelength_set_m,
        "ref_field": ref_field,
        "hold_field_zero_order": hold_field_zero_order,
    }
    save_data_path = savepath + "tfrcwav2_pml_350U_600H_" + str(FM) + "b.pickle"
    with open(save_data_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def nanofin_SweepFM():
    savepath = "physical_optical_layer/dev_testing/threeLayer_"
    call_library_generation_nanoFin(savepath, 11)
    return


if __name__ == "__main__":
    with tf.device("/cpu:0"):
        nanofin_SweepFM()


# # Generate SuperEllipse shape
# val = (
#     np.abs(2 * x_mesh / paramlist[i, 0]) ** paramlist[i, 2]
#     + np.abs(2 * y_mesh / paramlist[i, 1]) ** paramlist[i, 2]
#     - 1
# )
# val[np.where(val >= 0)] = 0.0
# val[np.where(val < 0)] = 1.0
# val = val[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
# ER_meta = 1 + (rcwa_parameters["erd"] - TF_ONE_COMPLEX) * val
