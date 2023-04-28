import tensorflow as tf
import numpy as np


def general_convolve(image, filter, rfft=False):
    """Runs the Fourier space convolution between an image and filter, where the filter kernels may have a different size from the image shape.

    Args:
        `image` (tf.float or tf.complex): Input image to apply the convolution filter kernel to, of shape [..., Ny, Nx]
        `filter` (tf.float or tf.complex): Convolutional filter kernel, of shape [..., My, Mx], but the same rank as the image input
        `rfft` (bool, optional): Flag to use real rfft instead of the general fft. Defaults to False.

    Returns:
        tf.float or tf.complex: Image with the filter convolved on the inner-most two dimensions
    """
    # Prepare the convolution by padding: Pad each image dimension equal to the initial filter shape
    init_image_shape = image.shape
    im_ny = init_image_shape[-2]
    im_nx = init_image_shape[-1]

    init_filter_shape = filter.shape
    filt_ny = init_filter_shape[-2]
    filt_nx = init_filter_shape[-1]

    # Zero pad the image with half the filter dimensionality and ensure the image is odd
    # padding = np.zeros((len(init_image_shape), 2))
    # padding[-2, :] = [filt_ny, filt_ny + 1] if np.mod(im_ny, 2) == 0 else [filt_ny, filt_ny]
    # padding[-1, :] = [filt_nx, filt_nx + 1] if np.mod(im_nx, 2) == 0 else [filt_nx, filt_nx]
    # image = tf.pad(image, padding)
    padding = np.zeros((len(init_image_shape), 2))
    if np.mod(im_nx, 2) == 0 and np.mod(filt_nx, 2) == 0:
        padding[-1, :] = [filt_nx // 2, filt_nx // 2 + 1]
    else:
        padding[-1, :] = [filt_nx // 2, filt_nx // 2]

    if np.mod(im_ny, 2) == 0 and np.mod(filt_ny, 2) == 0:
        padding[-2, :] = [filt_ny // 2, filt_ny // 2 + 1]
    else:
        padding[-2, :] = [filt_ny // 2, filt_ny // 2]
    image = tf.pad(image, padding)

    ### Pad the psf to match the new image dimensionality
    image_shape = tf.constant(image.shape)
    filter_shape = tf.constant(filter.shape)
    filter_resh = tf.reshape(
        tf.image.resize_with_crop_or_pad(tf.reshape(filter, [-1, filter_shape[-2], filter_shape[-1], 1]), image_shape[-2], image_shape[-1]),
        tf.concat([filter_shape[:-2], image_shape[-2:]], axis=0),
    )

    ### Run the convolution
    convolve_func = fourier_convolve_real if rfft else fourier_convolve
    new_image = tf.math.real(convolve_func(image, filter_resh))
    ### Undo odd padding if it was done before FFT
    new_image_shape = new_image.shape
    return tf.reshape(
        tf.image.resize_with_crop_or_pad(
            tf.reshape(new_image, [-1, image_shape[-2], image_shape[-1], 1]),
            init_image_shape[-2],
            init_image_shape[-1],
        ),
        # tf.concat([filter_shape[:-2], init_image_shape[-2:]], axis=0),
        tf.concat([new_image_shape[:-2], init_image_shape[-2:]], axis=0),
    )


def fourier_convolve(image, filter):
    """Computes the convolution of two signals (real or complex) using frequency space multiplcation. Convolution is done over the two inner-most dimensions.

    Args:
        `image` (tf.float or tf.complex): Image to apply filter to, of shape [..., Ny, Nx]
        `filter` (tf.float or tf.complex): Filter kernel; The kernel must be the same shape as the image

    Returns:
        tf.complex: Image with filter convolved, same shape as input
    """
    TF_ZERO = tf.cast(0.0, image.dtype)
    fourier_product = tf.signal.fft2d(tf.signal.ifftshift(tf.complex(image, TF_ZERO))) * tf.signal.fft2d(tf.signal.ifftshift(tf.complex(filter, TF_ZERO)))

    return tf.signal.fftshift(tf.signal.ifft2d(fourier_product))


def fourier_convolve_real(image, filter):
    """Computes the convolution of two, real-valued signals using frequency space multiplication. Convolution is done over the two inner-most dimensions.

    Args:
        image (tf.float): Image to apply filter to, of shape [..., Ny, Nx]
        filter (tf.float):  Filter kernel; The kernel must be the same shape as the image

    Returns:
        tf.float: Image with filter convolved, same shape as input
    """
    TF_ZERO = tf.cast(0.0, filter.dtype)
    fft_length = image.shape[-1]

    fourier_product = tf.signal.rfft2d(tf.signal.ifftshift(image)) * tf.signal.rfft2d(tf.signal.ifftshift(filter))
    return tf.signal.fftshift(tf.signal.irfft2d(fourier_product, [fft_length, fft_length]))
