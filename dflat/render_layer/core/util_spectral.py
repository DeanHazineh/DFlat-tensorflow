from scipy import interpolate
from pathlib import Path
import numpy as np
import tensorflow as tf


def get_path_to_data(file_name: str):
    resource_path = Path(__file__).parent / ""
    return resource_path.joinpath(file_name)


def get_xyz_bar_CIE1931(channels_nm):
    xyzpath = get_path_to_data("color_space/1931XYZFunctions.txt")
    xyzCIE = np.loadtxt(xyzpath, skiprows=1)

    xbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 1], kind="linear")
    ybar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 2], kind="linear")
    zbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 3], kind="linear")

    return np.transpose(np.stack([xbar_(channels_nm), ybar_(channels_nm), zbar_(channels_nm)], axis=0))


def get_rgb_bar_CIE1931(channels_nm):

    channels_nm = np.squeeze(channels_nm)
    rgbpath = get_path_to_data("color_space/1931RGBFunctions.txt")
    rgbCIE = np.loadtxt(rgbpath, skiprows=1)

    rbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 1], kind="linear")
    gbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 2], kind="linear")
    bbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 3], kind="linear")

    return np.transpose(np.stack([rbar_(channels_nm), gbar_(channels_nm), bbar_(channels_nm)], axis=0))


def get_illuminant_6500(channels_nm):
    channels_nm = channels_nm.flatten()
    illum_path = get_path_to_data("illuminants/D65.txt")
    illum = np.loadtxt(illum_path, skiprows=1)

    illum_ = interpolate.interp1d(illum[:, 0], illum[:, 1], kind="linear")

    return illum_(channels_nm)


def hsi_to_rgb(hsi_cube, wavelength_set_nm):
    # hsi_cube has wl in the last dimension
    cmf_bar = get_rgb_bar_CIE1931(wavelength_set_nm)

    return tf.linalg.matmul(hsi_cube, cmf_bar)
