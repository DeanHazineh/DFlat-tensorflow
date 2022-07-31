import h5py
from scipy.interpolate import interp1d
import numpy as np

MATERIAL_DICT = {
    "TiO2": "dflat/physical_optical_layer/core/material_index/TiO2_Index.mat",
    "SiO2": "dflat/physical_optical_layer/core/material_index/SiO2_Index.mat",
    "Vacuum": None,
}


def get_material_index(material_name, wavelength_list):
    # Scipy interp1d function allows for complex numbers

    index_path = MATERIAL_DICT[material_name]
    if material_name == "Vacuum":
        return (1 + 1j * 0) * np.ones(shape=(len(wavelength_list)))

    else:
        data = {}
        f = h5py.File(index_path)
        for k, v in f.items():
            data[k] = np.array(v)

        index_dat = data["index"]
        index_dat = np.squeeze(index_dat["real"] + 1j * index_dat["imag"])
        wavelength_dat = np.squeeze(data["w"])

        if (np.min(wavelength_list) < np.min(wavelength_dat)) or (np.max(wavelength_list) > np.max(wavelength_dat)):
            raise ValueError("get_material_index: wavelength is outside the boundaries of the index dat file")
        else:
            interp_func = interp1d(wavelength_dat, index_dat)

    return interp_func(wavelength_list)
