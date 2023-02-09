import tensorflow as tf


def photons_to_ADU(image_photons, sensor_parameters):
    dtype = image_photons.dtype

    ### Shot Noise on electron signal
    QE = sensor_parameters["QE"]
    electrons_signal = image_photons * QE

    if sensor_parameters["shot_noise"]:
        poisson_noise = tf.stop_gradient(tf.cast(tf.experimental.numpy.random.poisson(electrons_signal), dtype) - electrons_signal)
        electrons_signal = electrons_signal + poisson_noise

    ### total dark noise electrons
    if sensor_parameters["dark_noise"]:
        electrons_dark = tf.random.normal(
            image_photons.shape,
            mean=sensor_parameters["dark_offset"],
            stddev=sensor_parameters["dark_noise_e"],
            dtype=dtype,
        )
    else:
        electrons_dark = tf.zeros_like(electrons_signal)

    # return tf.math.round(sensor_parameters["gain"] * (electrons_signal + electrons_dark))
    return sensor_parameters["gain"] * (electrons_signal + electrons_dark)
