import tensorflow as tf
import numpy as np
from .hankel import iqdht, qdht, tf_generalSpline_regular1DGrid


def transfer_function_diffraction(
    wavefront_ampl,
    wavefront_phase,
    wavelength_m,
    distance_m,
    input_pixel_size_m,
    input_pixel_number,
    output_pixel_size_m,
    dtype,
    radial_symmetry,
    optArg=1,
):
    """Uses the angular spectrum method to propagate an input complex field to the output plane. 

    Args:
        `wavefront_ampl` (tf.float): Starting field amplitude, of shape (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r'])
        `wavefront_phase` (tf.float): Starting field phase, the same shape as wavefront_ampl.
        `wavelength_m` (tf.float): Tf constant defining the wavelength of light for the calculation, in units of m
        `distance_m` (tf.float64): Tf constant defining the distance between the starting plane and the propagated plane, in units of m
        `input_pixel_size_m` (dict): Starting field grid discretization/pitch in units of m, via dictionary {"x": float, "y": float}. 
        `input_pixel_number` (dict): Starting field grid size, in terms of number of pixels, via dictionary {"x": float, "y": float}.
        `output_pixel_size_m` (dict): Unused but kept as input to match the fresnel_diffraction_fft method input arguments.
        `dtype` (tf.dtype): Datatype for the calculation. Only tf.float64 is currently allowed
        `radial_symmetry` (bool): Flag indicating if radial symmetry is used.
        `optArg` (int, optional): Defines an additional length factor for zero-padding the radial data, to be used when
            conducting frequency space transforms. Defaults to 0.

    Returns:
        `tf.float64`: Field amplitude at the output plane grid, of shape 
            (batch_size, input_pixel_number['y'], input_pixel_number['x']) or (batch_size, 1, input_pixel_number['r'])
        `tf.float64`: Field phase at the output plane grid, of shape
            (batch_size, input_pixel_number['y'], input_pixel_number['x']) or (batch_size, 1, input_pixel_number['r'])

    """

    TF_ZERO = tf.cast(0.0, dtype=dtype)

    ### Enable padding for frequency transforms if requested
    padFactor = optArg
    padhalfx = tf.cast(input_pixel_number["x"] * padFactor, tf.int32)
    padhalfy = tf.cast(input_pixel_number["y"] * padFactor, tf.int32)
    paddings = tf.cond(
        radial_symmetry,
        lambda: [[0, 0], [0, 0], [0, padhalfx]],
        lambda: [[0, 0], [padhalfy, padhalfy], [padhalfx, padhalfx]],
    )
    padded_wavefront_ampl = tf.pad(wavefront_ampl, paddings, mode="CONSTANT", constant_values=0)
    padded_wavefront_phase = tf.pad(wavefront_phase, paddings, mode="CONSTANT", constant_values=0)

    ### Define the grid space (input same as output spatial domain)
    if radial_symmetry:
        x, y = tf.meshgrid(tf.range(0, input_pixel_number["r"] + padhalfx, dtype=dtype), tf.range(1, dtype=dtype))
    else:
        x, y = tf.meshgrid(
            tf.range(0, input_pixel_number["x"] + 2 * padhalfx, dtype=dtype),
            tf.range(0, input_pixel_number["y"] + 2 * padhalfy, dtype=dtype),
        )
        x = x - (x.shape[1] - 1) / 2
        y = y - (y.shape[0] - 1) / 2
    x = x * input_pixel_size_m["x"]
    y = y * input_pixel_size_m["y"]

    ### Get the angular decomposition of the input field
    fourier_transform_term = tf.complex(padded_wavefront_ampl, TF_ZERO) * tf.exp(
        tf.complex(TF_ZERO, padded_wavefront_phase)
    )
    if radial_symmetry:
        kr, angular_spectrum = qdht(tf.squeeze(x), fourier_transform_term)
    else:
        angular_spectrum = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(fourier_transform_term)))

    #### Define the transfer function via FT of the Sommerfield solution
    # Define the output field grid
    rarray = tf.math.sqrt(distance_m ** 2 + x ** 2 + y ** 2)
    angular_wavenumber = tf.cast(2 * np.pi / wavelength_m, dtype=dtype)
    h = tf.expand_dims(
        (
            tf.complex(1 / 2 / np.pi * distance_m / rarray ** 2, TF_ZERO)
            * tf.complex(1 / rarray, -1 * angular_wavenumber)
            * tf.exp(tf.complex(TF_ZERO, angular_wavenumber * rarray))
        ),
        0,
    )

    # Compute Fourier Space Transfer Function
    if radial_symmetry:
        kr, H = qdht(tf.squeeze(x), h)
    else:
        H = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(h)))

    # note: we have decided to ingore the physical, depth dependent energy scaling here (and in the fresnel method)
    # This change makes it easier to play with normalized PSF (Energy under the IPSF less than or equal to energy incident on aperture)
    H = tf.exp(tf.complex(TF_ZERO, tf.math.angle(H)))

    ### Propagation by multiplying angular decomposition with H then taking the inverse transform
    fourier_transform_term = angular_spectrum * H
    if radial_symmetry:
        r2, outputwavefront = iqdht(kr, fourier_transform_term)
        outputwavefront = tf_generalSpline_regular1DGrid(r2, tf.squeeze(x), outputwavefront)
    else:
        outputwavefront = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(fourier_transform_term)))

    ### Crop to remove the padding used in the calculation
    # Radial symmetry needs special preperation before cropping because cropping
    # runs a inner crop so we need to add half padding for symmetry then call crop
    outputwavefront = tf.cond(
        radial_symmetry,
        lambda: tf.pad(outputwavefront, [[0, 0], [0, 0], [padhalfx, 0]], mode="CONSTANT"),
        lambda: outputwavefront,
    )

    outputwavefront = tf.image.resize_with_crop_or_pad(
        tf.expand_dims(outputwavefront, -1),  # need to make it shape [N, nx, ny, channels=1]
        tf.cond(radial_symmetry, lambda: 1, lambda: input_pixel_number["y"]),
        tf.cond(radial_symmetry, lambda: input_pixel_number["r"], lambda: input_pixel_number["x"],),
    )

    outputwavefront_ampl = tf.math.abs(outputwavefront)
    outputwavefront_phase = tf.math.angle(outputwavefront)

    return tf.squeeze(outputwavefront_ampl, -1), tf.squeeze(outputwavefront_phase, -1)
