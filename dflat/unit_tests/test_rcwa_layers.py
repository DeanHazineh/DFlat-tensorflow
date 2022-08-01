import tensorflow as tf
from dflat.physical_optical_layer.core.ms_parameterization import ALLOWED_PARAMETERIZATION_TYPE
import dflat.data_structure as df_struct
import dflat.physical_optical_layer as df_physical

wavelength_set_m = [450e-9, 600e-9]
FM = 5
rcwa_settings = {
    "wavelength_set_m": wavelength_set_m,
    "thetas": [0.0 for i in wavelength_set_m],
    "phis": [0.0 for i in wavelength_set_m],
    "pte": [1.0 for i in wavelength_set_m],
    "ptm": [1.0 for i in wavelength_set_m],
    "pixelsX": 2,
    "pixelsY": 2,
    "PQ": [FM, FM],
    "Lx": 350e-9,
    "Ly": 350e-9,
    "L": [1e-3, 600.0e-9, 1e-3],
    "Lay_mat": ["SiO2", "Vacuum", "Vacuum"],
    "material_dielectric": "TiO2",
    "er1": "Vacuum",
    "er2": "Vacuum",
    "Nx": 512,
    "Ny": 512,
    "parameterization_type": None,
    "batch_wavelength_dim": False,
}


def test_rcwa_layers():
    rcwa_settings["batch_wavelength_dim"] = False

    for parameterization in ALLOWED_PARAMETERIZATION_TYPE.keys():
        if parameterization != "None":
            rcwa_settings["parameterization_type"] = parameterization
            rcwa_parameters = df_struct.rcwa_params(rcwa_settings)

        # Check run rcwa_layer
        rcwa_layer = df_physical.RCWA_Layer(rcwa_parameters)
        shape_vect = tf.ones(rcwa_layer.shape_vect_size)
        rcwa_layer(shape_vect)

        # Check run for rcwa_latent_layer
        rcwa_latent_layer = df_physical.RCWA_Layer(rcwa_parameters)
        rcwa_latent_layer(shape_vect)

    return 1


def test_rcwa_layers_batched():
    rcwa_settings["batch_wavelength_dim"] = True

    for parameterization in ALLOWED_PARAMETERIZATION_TYPE.keys():
        if parameterization != "None":
            rcwa_settings["parameterization_type"] = parameterization
            rcwa_parameters = df_struct.rcwa_params(rcwa_settings)

        # Check run rcwa_layer
        rcwa_layer = df_physical.RCWA_Layer(rcwa_parameters)
        shape_vect = tf.ones(rcwa_layer.shape_vect_size)
        rcwa_layer(shape_vect)

        # Check run for rcwa_latent_layer
        rcwa_latent_layer = df_physical.RCWA_Layer(rcwa_parameters)
        rcwa_latent_layer(shape_vect)

    return 1

