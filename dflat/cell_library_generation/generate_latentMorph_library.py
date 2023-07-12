import numpy as np
import tensorflow as tf

import dflat.data_structure as df_struct
from dflat.cell_library_generation import run_library_gen, assemble_latentMorph_freeform


def gen_freeform_library(savepath, fourier_modes=11, num_rect=4, num_sim=10, checkpoint_num=100):
    wavelength_set_m = np.arange(350e-9, 750e-9, 5e-9)
    constraint_dict = {"brule": 100e-9, "fm": 60e-9, "Ux": 600e-9, "Uy": 600e-9, "Nx": 600, "Ny": 600, "constraint_type": "morphological"}
    rcwa_parameters = df_struct.rcwa_params(
        {
            "wavelength_set_m": wavelength_set_m,
            "thetas": [0.0 for i in wavelength_set_m],
            "phis": [0.0 for i in wavelength_set_m],
            "pte": [1.0 for i in wavelength_set_m],
            "ptm": [1.0 for i in wavelength_set_m],
            "pixelsX": 1,
            "pixelsY": 1,
            "PQ": [fourier_modes, fourier_modes],
            "Lx": constraint_dict["Ux"],
            "Ly": constraint_dict["Ux"],
            "L": [600e-9],
            "Lay_mat": ["Vacuum"],
            "material_dielectric": "TiO2",
            "er1": "SiO2",
            "er2": "Vacuum",
            "Nx": constraint_dict["Nx"],
            "Ny": constraint_dict["Ny"],
            "dtype": tf.float32,
            "cdtype": tf.complex64,
        }
    )

    param_list = np.random.random(size=(num_sim, 5 * num_rect))
    run_library_gen(
        rcwa_parameters, param_list, assemble_latentMorph_freeform, fun_args=[constraint_dict], showDebugPlot=False, savepath=savepath, checkpoint_num=checkpoint_num, zero_order_only=False
    )

    return


if __name__ == "__main__":
    # Repeat the library gen for batches
    for i in range(100):
        savepath = f"dflat/cell_library_generation/output/Freeform4_batch_{i}_"
        gen_freeform_library(savepath, fourier_modes=11, num_rect=4, num_sim=500)
