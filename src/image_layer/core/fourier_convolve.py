import tensorflow as tf


def fourier_convolve(pinholeImage, filter):
    # pinholeImage should have shape [, Ny, Nx] or [1, Ny, Nx], eg fft taken over inner two dimensions
    # filters (e.g. psf) should have shape [, Ny, Nx]
    # fft2d computes fourier transform on the innermost two dimensions
    TF_ZERO = tf.cast(0.0, filter.dtype)

    fourier_product = tf.signal.fftshift(
        tf.signal.fft2d(tf.signal.ifftshift(tf.complex(pinholeImage, TF_ZERO)))
    ) * tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(tf.complex(filter, TF_ZERO))))

    return tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(fourier_product)))


def fourier_convolve_real(pinholeImage, filter):

    return tf.math.abs(fourier_convolve(pinholeImage, filter))
