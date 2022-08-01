import numpy as np

import dflat.fourier_layer as df_fourier
import dflat.data_structure as df_struct
from dflat.data_structure.fourier_params import diffractionEngines

radial_Flag = [True, False]
zs = [1e-3, 0.5e-3, 0.1e-3]
point_source_locs = np.array([[0.0, 0.0, z] for z in zs])

propagation_parameters_mono = {
    "wavelength_m": 532e-9,
    "ms_length_m": {"x": 0.1e-3, "y": 0.1e-3},
    "ms_dx_m": {"x": 3e-6, "y": 3e-6},
    "radius_m": 2e-3,
    "sensor_distance_m": 10e-3,
    "initial_sensor_dx_m": {"x": 3e-6, "y": 3e-6},
    "sensor_pixel_size_m": {"x": 3e-6, "y": 3e-6},
    "sensor_pixel_number": {"x": 101, "y": 101},
    "radial_symmetry": True,
    "diffractionEngine": "fresnel_fourier",
    "accurate_measurement": False,  # Flag ensures output grid is exact but is expensive
}


propagation_parameters_broadband = {
    "wavelength_set_m": [500e-9, 600e-9],
    "ms_length_m": {"x": 0.1e-3, "y": 0.1e-3},
    "ms_dx_m": {"x": 3e-6, "y": 3e-6},
    "radius_m": None,
    "sensor_distance_m": 10e-3,
    "initial_sensor_dx_m": {"x": 3e-6, "y": 3e-6},
    "sensor_pixel_size_m": {"x": 3e-6, "y": 3e-6},
    "sensor_pixel_number": {"x": 101, "y": 101},
    "radial_symmetry": True,
    "diffractionEngine": "fresnel_fourier",
    "accurate_measurement": False,  # Flag ensures output grid is exact but is expensive
}


def test_fourier_layers_mono():

    for engine in diffractionEngines:
        for radial_symmetry in radial_Flag:

            propagation_parameters_mono["radial_symmetry"] = radial_symmetry
            propagation_parameters_mono["diffractionEngine"] = engine
            propagation_parameters = df_struct.prop_params(propagation_parameters_mono)

            #
            gridshape = propagation_parameters["grid_shape"]
            batch_size = 2
            gridshape[0] = batch_size
            field_input = [np.ones(gridshape), np.ones(gridshape)]

            Propagate_planes_layer_mono = df_fourier.Propagate_Planes_Layer_Mono(propagation_parameters)
            out = Propagate_planes_layer_mono(field_input)

            #
            PSF_Layer_Mono = df_fourier.PSF_Layer_Mono(propagation_parameters)
            out = PSF_Layer_Mono(field_input, point_source_locs)

            alternative_input = [np.expand_dims(field_input[0], 0), np.expand_dims(field_input[1], 0)]
            out = PSF_Layer_Mono(alternative_input, point_source_locs)

    return


def test_fourier_layers_broadband():

    for engine in diffractionEngines:
        for radial_symmetry in radial_Flag:

            propagation_parameters_broadband["radial_symmetry"] = radial_symmetry
            propagation_parameters_broadband["diffractionEngine"] = engine
            propagation_parameters = df_struct.prop_params(propagation_parameters_broadband)

            wavelength_list = propagation_parameters["wavelength_set_m"]
            gridshape = propagation_parameters["grid_shape"]
            batch_size = 2
            field1 = np.ones((len(wavelength_list), batch_size, gridshape[1], gridshape[2]))
            field2 = np.ones((batch_size, gridshape[1], gridshape[2]))

            #
            propagate_planes_layer = df_fourier.Propagate_Planes_Layer(propagation_parameters)
            out = propagate_planes_layer([field1, field1])
            out = propagate_planes_layer([field2, field2])

            #
            psf_layer = df_fourier.PSF_Layer(propagation_parameters)
            out = psf_layer([field1, field1], point_source_locs)
            out = psf_layer([field2, field2], point_source_locs)

    return


# if __name__ == "__main__":
#     print("temporary main")
#     test_fourier_layers_broadband()
