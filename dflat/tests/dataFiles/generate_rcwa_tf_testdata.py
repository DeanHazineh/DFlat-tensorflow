import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

import dflat.cell_library_generation as lib_gen
import dflat.data_structure as df_struct
import dflat.tools as df_tools


def run_nanofin_broadband_gen(FM=9):

    ### Specify RCWA Solver parameters
    wavelength_set_m = np.arange(400e-9, 705e-9, 5e-9)
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
        "L": [100e-6, 600.0e-9, 1e-3],
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
    rcwa_parameters = df_struct.rcwa_params(rcwa_settings)

    ### Define sweep ranges and savepath
    len_x = [100e-9, 200e-9]
    len_y = [100e-9, 200e-9]
    Len_x, Len_y = np.meshgrid(len_x, len_y)
    paramlist = np.transpose(np.vstack((Len_x.flatten(), Len_y.flatten())))
    savepath = "dflat/tests/dataFiles/rcwa_tf_broadbandFins_"

    ### Run library Sweep
    ref_field, hold_field_zero_order = lib_gen.run_zeroOrder_library_gen(
        rcwa_parameters, paramlist, cell_fun=lib_gen.assemble_ER_rectangular_fin, showDebugPlot=False
    )

    trans = np.abs(hold_field_zero_order) ** 2
    phase = np.angle(hold_field_zero_order) - np.angle(ref_field)
    trans = trans.reshape([len(len_y), len(len_x), len(wavelength_set_m), 2])
    phase = phase.reshape([len(len_y), len(len_x), len(wavelength_set_m), 2])

    ### Save the data
    data = {
        "trans": trans,
        "phase": phase,
        "paramlist": paramlist,
        "wavelength_set_m": wavelength_set_m,
        "ref_field": ref_field,
        "hold_field_zero_order": hold_field_zero_order,
    }
    save_data_path = savepath + "FM" + str(FM) + "b.pickle"
    with open(save_data_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":

    with tf.device("/cpu:0"):  # comment to run on gpu
        run_nanofin_broadband_gen(FM=9)
