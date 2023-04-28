import tensorflow as tf
from .util_spectral import get_rgb_bar_CIE1931


def photons_to_ADU(image_photons, sensor_parameters, clip_zero=True):
    dtype = image_photons.dtype

    ### Shot Noise on electron signal
    electrons_signal = image_photons * sensor_parameters["QE"]

    if sensor_parameters["shot_noise"]:
        poisson_noise = tf.stop_gradient(tf.cast(tf.experimental.numpy.random.poisson(electrons_signal), dtype) - electrons_signal)
        electrons_signal = electrons_signal + poisson_noise

    ### total dark noise electrons
    if sensor_parameters["dark_noise"]:
        electrons_dark = tf.random.normal(image_photons.shape, mean=sensor_parameters["dark_offset"], stddev=sensor_parameters["dark_noise_e"], dtype=dtype)
    else:
        electrons_dark = tf.zeros_like(electrons_signal)

    if clip_zero:
        return tf.clip_by_value(sensor_parameters["gain"] * (electrons_signal + electrons_dark), 0, tf.float64.max)
    else:
        return sensor_parameters["gain"] * (electrons_signal + electrons_dark)


def hsi_to_rgb(hsi_cube, wavelength_set_nm):
    # hsi_cube has wl in the last dimension
    cmf_bar = get_rgb_bar_CIE1931(wavelength_set_nm)

    return tf.linalg.matmul(hsi_cube, cmf_bar)
