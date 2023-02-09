import tensorflow as tf

from .DEPRECATED_sensor_functions import get_rgb_bar_CIE1931

# A good resource to read if you look at this part and want to learn what is happening
# https://medium.com/hipster-color-science/a-beginners-guide-to-colorimetry-401f1830b65a
# https://medium.com/hipster-color-science/color-reproductions-of-hyperspectral-images-ad6210bbcd1d
# https://personalpages.manchester.ac.uk/staff/d.h.foster/Tutorial_HSI2RGB/Tutorial_HSI2RGB.html


def hsi_to_rgb(hsi_cube, wavelength_set_nm):
    # hsi_cube has wl in the last dimension
    cmf_bar = get_rgb_bar_CIE1931(wavelength_set_nm)

    return tf.linalg.matmul(hsi_cube, cmf_bar)


# def refl_spectral_to_rgb(hs_cube, channels_nm, illuminant):

#     # # Using XYZ Functions
#     # cmf_bar = get_xyz_bar_CIE1931(channels_nm)
#     # # K = np.sum(illuminant * cmf_bar[:, 1])
#     # # tristimulus = np.matmul(hs_cube, cmf_bar) / K
#     # # tristimulus = tristimulus / np.max(tristimulus)
#     # # tri_shape = tristimulus.shape
#     # # RGB = np.matmul(np.reshape(tristimulus, [tri_shape[0] * tri_shape[1], tri_shape[2]]), M_XYZ_to_srgb)
#     # # RGB = np.reshape(RGB, [tri_shape[0], tri_shape[1], tri_shape[2]])

#     cmf_bar = get_rgb_bar_CIE1931(channels_nm)
#     hs_cube = hs_cube / illuminant
#     RGB = np.matmul(hs_cube, cmf_bar)
#     RGB = RGB / np.max(RGB)
#     # N = np.sum(illuminant * cmf_bar[:, 1:2])
#     # RGB = np.matmul(hs_cube, cmf_bar) / N

#     return RGB
