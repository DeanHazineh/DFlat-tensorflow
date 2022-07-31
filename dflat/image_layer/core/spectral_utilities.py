import numpy as np
from scipy import interpolate
from .color_space.color_space_conversion import *


def get_xyz_bar_CIE1931(channels_nm):

    xyzpath = "dflat/image_layer/core/color_space/1931XYZFunctions.txt"
    xyzCIE = np.loadtxt(xyzpath, skiprows=1)

    xbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 1], kind="linear")
    ybar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 2], kind="linear")
    zbar_ = interpolate.interp1d(xyzCIE[:, 0], xyzCIE[:, 3], kind="linear")

    return np.transpose(np.squeeze(np.stack([xbar_(channels_nm), ybar_(channels_nm), zbar_(channels_nm)], axis=0), 1))


def get_rgb_bar_CIE1931(channels_nm):

    channels_nm = channels_nm.flatten()
    rgbpath = "dflat/image_layer/core/color_space/1931RGBFunctions.txt"
    rgbCIE = np.loadtxt(rgbpath, skiprows=1)

    rbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 1], kind="linear")
    gbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 2], kind="linear")
    bbar_ = interpolate.interp1d(rgbCIE[:, 0], rgbCIE[:, 3], kind="linear")

    return np.transpose(np.stack([rbar_(channels_nm), gbar_(channels_nm), bbar_(channels_nm)], axis=0))


def refl_spectral_to_rgb(hs_cube, channels_nm, illuminant):
    """_summary_

    Args:
        hs_cube (float): Hyperspectral data cube, of shape (Ny, Nx, channels)
        channels_nm (float): Uniform spaced, wavelength values in nm for spectral channels
        illuminant (float): Illumination source spectral power, same length as channels_nm
    """

    rgb_bar = get_rgb_bar_CIE1931(channels_nm)
    N = np.sum(illuminant * rgb_bar[:, 1:2])
    RGB = np.matmul(hs_cube, rgb_bar) / N
    return RGB
