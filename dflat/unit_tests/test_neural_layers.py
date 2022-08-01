import numpy as np
import dflat.neural_optical_layer as df_neural
from dflat.datasets_metasurface_cells.libraryClass import listLibraryNames, polarization_basis

model_names = ["MLP_Nanocylinders_Dense32_U180_H600", "MLP_Nanofins_Dense32_U350_H600"]
wavelength_list = [400e-9, 500e-9]
gridshape = [1, 2, 2]


def test_neural_layer():
    for name in model_names:
        # Call latent layer functions
        MLP_latent_layer = df_neural.MLP_Latent_Layer(name)
        input_tensor = MLP_latent_layer.initialize_input_tensor("uniform", gridshape)
        unnorm_shape = MLP_latent_layer.latent_to_unnorm_shape(input_tensor)
        MLP_latent_layer(input_tensor, wavelength_list)

        # Call MLP layer
        MLP_layer = df_neural.MLP_Layer(name)
        input_tensor = MLP_layer.initialize_input_tensor("uniform", gridshape)
        unnorm_shape = MLP_layer.param_to_unnorm_shape(input_tensor)
        MLP_layer(input_tensor, wavelength_list)

    return 1


def test_response_to_param():
    num_profiles = 2
    for idx, libname in enumerate(listLibraryNames):
        ms_trans_aslist = [np.ones((polarization_basis[idx], 2, 2)) for i_prof in range(num_profiles)]
        ms_phase_aslist = [np.ones((polarization_basis[idx], 2, 2)) for i_prof in range(num_profiles)]

        df_neural.optical_response_to_param(ms_trans_aslist, ms_phase_aslist, wavelength_list, libname, reshape=False)
        df_neural.optical_response_to_param(ms_trans_aslist, ms_phase_aslist, wavelength_list, libname, reshape=True)

    return
