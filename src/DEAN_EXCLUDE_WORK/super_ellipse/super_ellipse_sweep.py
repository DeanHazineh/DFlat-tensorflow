import sys
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

sys.path.append(".")
from cell_library_generation.core.assemble_ER import assemble_ER_superEllipse
from cell_library_generation.core.run_sweep_call import run_zeroOrder_library_gen
from data_structure import rcwa_params as rcwa_params
import tools.graphFunc as gF


def run_superEllipse_Sweep(FM=11):

    ### Specify RCWA Solver parameters
    wavelength_set_m = [450e-9, 500e-9, 550e-9, 600e-9]
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
        "er1": "Vacuum",
        "er2": "Vacuum",
        "Nx": 1024,
        "Ny": 1024,
        "parameterization_type": "None",
        "batch_wavelength_dim": False,
        "dtype": tf.float32,
        "cdtype": tf.complex64,
    }
    rcwa_parameters = rcwa_params(rcwa_settings, bare=True)

    ### Define sweep ranges and savepath
    len_x = np.arange(60e-9, 300e-9, 5e-9)
    sweep_n_values = np.arange(0.3, 7, 0.2)
    savepath = "cell_library_generation/dev_test/output/super_ellipse/sweep2/"

    for n_val in sweep_n_values:
        n_vec = n_val * np.ones_like(len_x)
        paramlist = np.transpose(np.vstack([len_x, len_x, n_vec]))
        ref_field, hold_field_zero_order = run_zeroOrder_library_gen(
            rcwa_parameters, paramlist, cell_fun=assemble_ER_superEllipse, showDebugPlot=False
        )

        trans = np.abs(hold_field_zero_order) ** 2
        ref_trans = np.abs(ref_field) ** 2
        phase = np.angle(hold_field_zero_order)
        ref_phase = np.angle(ref_field)

        # trans = trans / ref_trans
        trans = trans
        phase = phase - ref_phase
        trans = trans.reshape([1, len(len_x), len(wavelength_set_m), 2])
        phase = phase.reshape([1, len(len_x), len(wavelength_set_m), 2])

        ### Save the data
        data = {
            "trans": trans,
            "phase": phase,
            "paramlist": paramlist,
            "wavelength_set_m": wavelength_set_m,
            "ref_field": ref_field,
            "hold_field_zero_order": hold_field_zero_order,
            "n_val": n_val,
        }
        save_data_path = savepath + "run_superEllipse_Sweep1D_n" + f"{n_val:2.1f}" + "_FM" + str(FM) + "b.pickle"
        with open(save_data_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    run_superEllipse_Sweep(FM=9)
