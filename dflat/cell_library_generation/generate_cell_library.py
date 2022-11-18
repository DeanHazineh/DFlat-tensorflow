import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

import dflat.cell_library_generation as lib_gen
import dflat.data_structure as df_struct
import dflat.tools as df_tools


def run_nanofin_Sweep(FM, shape_function):

    ### Specify RCWA Solver parameters
    wavelength_set_m = np.arange(310e-9, 750e-9, 1e-9)
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
        "L": [600.0e-9],
        "Lay_mat": ["Vacuum"],
        "layer_dielectric": 1,  # which layer in the list above to pattern on, counting from 1
        "material_dielectric": "TiO2",
        "er1": "SiO2",
        "er2": "Vacuum",
        "Nx": 512,
        "Ny": 512,
        "parameterization_type": "None",  # Parameter not used for this forward design code
        "batch_wavelength_dim": False,  # Library generation will be very slow if you batch! lib_gen not allowing it!
        "dtype": tf.float32,
        "cdtype": tf.complex64,
    }
    rcwa_parameters = df_struct.rcwa_params(rcwa_settings)

    ### Define sweep ranges and savepath
    # Rotation can be computed more efficiently by basis change on eigenvectors but sometimes, I actually want the values
    len_x = np.arange(60e-9, 305e-9, 5e-9)
    len_y = np.arange(60e-9, 305e-9, 5e-9)
    theta = [0.0]
    Len_x, Len_y, Theta = np.meshgrid(len_x, len_y, theta)
    paramlist = np.transpose(np.vstack((Len_x.flatten(), Len_y.flatten(), Theta.flatten())))

    ### Run library Sweep
    savepath = "dflat/cell_library_generation/output/nanofin_holes"
    transmission, phase = lib_gen.run_zeroOrder_library_gen(
        rcwa_parameters,
        paramlist,
        cell_fun=shape_function,
        showDebugPlot=False,
        savepath=savepath,
        checkpoint_num=250,
    )

    ### Save the data
    data = {
        "transmission": transmission,
        "phase": phase,
        "paramlist": paramlist,
        "lenx": len_x,
        "leny": len_y,
        "theta": theta,
        "wavelength_set_m": wavelength_set_m,
    }
    save_data_path = savepath + "nanofin_library.pickle"
    with open(save_data_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    # run_nanofin_Sweep(FM=9, shape_function=lib_gen.assemble_ER_rectangular_fin)
    run_nanofin_Sweep(FM=9, shape_function=lib_gen.assemble_ER_inverse_rectangular_resonator)
