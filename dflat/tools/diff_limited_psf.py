import numpy as np
from scipy.special import j1
from dflat.fourier_layer.core.ops_hankel import radial_2d_transform


def airy_disk(wavelength_m, radius_m, xgrid_vect, sensor_distance_m):
    if len(xgrid_vect.shape) == 2:
        xgrid_vect = np.squeeze(xgrid_vect)

    xin = 2 * np.pi / wavelength_m * radius_m * np.sin(xgrid_vect / np.sqrt(sensor_distance_m**2 + xgrid_vect**2)) + 1e-6
    airyprof = (2 * j1(xin) / xin) ** 2
    cidx = int((len(xgrid_vect) - 1) / 2 + 1)

    airyprof = radial_2d_transform(np.squeeze(airyprof[cidx - 1 :]))
    airyprof = airyprof / np.sum(airyprof)

    return airyprof
