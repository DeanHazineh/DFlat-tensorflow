import tensorflow as tf
import numpy as np

from dflat.cell_library_generation import run_library_gen, assemble_ER_super_ellipse
import dflat.data_structure as df_struct


def gen_super_ellipse_library(savepath, fourier_modes=9, checkpoint_num=100):
    wavelength_set_m = np.arange(350e-9, 750e-9, 5e-9)
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
            "Lx": 350e-9,
            "Ly": 350e-9,
            "L": [600.0e-9],
            "Lay_mat": ["Vacuum"],
            "material_dielectric": "TiO2",
            "er1": "SiO2",
            "er2": "Vacuum",
            "Nx": 512,
            "Ny": 512,
            "batch_wavelength_dim": False,  # Library generation will be very slow if you batch! lib_gen not allowing it!
            "dtype": tf.float32,
            "cdtype": tf.complex64,
        }
    )

    ## Define sweep ranges and savepath
    # Rotation can be computed more efficiently by basis change on eigenvectors
    # but sometimes, one actually wants to compute rotation directly (not here)
    len_x = np.arange(60e-9, 305e-9, 5e-9)
    len_y = np.arange(60e-9, 305e-9, 5e-9)
    theta = [0.0]
    Len_x, Len_y, Theta = np.meshgrid(len_x, len_y, theta)
    paramlist = np.transpose(np.vstack((Len_x.flatten(), Len_y.flatten(), Theta.flatten())))
    superEllipse_args = [20, False]  # exponential power and inverse True or False

    run_library_gen(
        rcwa_parameters,
        paramlist,
        assemble_ER_super_ellipse,
        fun_args=superEllipse_args,
        showDebugPlot=True,
        savepath=savepath,
        checkpoint_num=checkpoint_num,
        zero_order_only=False,
    )

    return


if __name__ == "__main__":
    savepath = "dflat/cell_library_generation/output/"
    gen_super_ellipse_library(savepath)
