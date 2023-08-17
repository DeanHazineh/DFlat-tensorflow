import tensorflow as tf
import numpy as np
from .ops_hankel import qdht
from .ops_transform_util import tf_generalSpline_regular1DGrid
from .ops_grid_util import tf_coordinate_grid


def fresnel_diffraction_fft(
    wavefront_ampl,
    wavefront_phase,
    wavelength_m,
    distance_m,
    input_pixel_size_m,
    input_pixel_number,
    output_pixel_size_m,
    dtype,
    radial_symmetry,
    optArg=0,
):
    """Uses the single-Fourier transform implementation of the Fresnel diffraction equation to propagate fields.

    The complex coefficients in the formulation are excluded here but can be added by appropriately calling
    fresnel_diffraction_coeffs().

    Args:
        `wavefront_ampl` (tf.float): Starting field amplitude, of shape (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r']).
        `wavefront_phase` (tf.float): Starting field phase, the same shape as wavefront_ampl.
        `wavelength_m` (tf.float): Tf constant defining the wavelength of light for the calculation, in units of m
        `distance_m` (tf.float): Tf constant defining the distance between the starting plane and the propagated plane, in units of m
        `input_pixel_size_m` (dict): Starting field grid discretization/pitch in units of m, via dictionary {"x": float, "y": float}.
        `input_pixel_number` (dict): Starting field grid size, in terms of number of pixels, via dictionary {"x": float, "y": float}.
        `output_pixel_size_m` (dict): Propagated field grid discretization/pitch, in units of m, via dictionary {"x": float, "y":float}.
        `dtype` (tf.dtype): Datatype for the calculation. Only tf.float64 is currently allowed
        `radial_symmetry` (bool): Flag indicating if radial symmetry is used.
        `optArg` (int, optional): Unused for this call.
    Returns:
        `tf.float64`: Field amplitude at the output plane grid, of shape (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r']).
        `tf.float64`: Field phase at the output plane grid (batch_size, input_pixel_number['y'], input_pixel_number['x'])
            or (batch_size, 1, input_pixel_number['r'])
    """

    # create the coordinate grid at the input
    input_pixel_x, input_pixel_y = tf_coordinate_grid(input_pixel_number, input_pixel_size_m, radial_symmetry, dtype)
    TF_ZERO = tf.constant(0.0, dtype=dtype)

    # fourier transform approximation of fresnel diffraction
    angular_wave_number = 2 * np.pi / wavelength_m
    quadratic_term = angular_wave_number / 2 / distance_m * (input_pixel_x**2 + input_pixel_y**2)
    fourier_transform_trans = wavefront_ampl
    fourier_transform_phase = wavefront_phase + tf.expand_dims(quadratic_term, 0)
    fourier_transform_term = tf.complex(fourier_transform_trans, TF_ZERO) * tf.exp(tf.complex(TF_ZERO, fourier_transform_phase))

    # If radialy symmetric input, then use the hankel transform otherwise use 2D DFT
    if radial_symmetry:
        kr, wavefront_outPlane = qdht(tf.squeeze(input_pixel_x), fourier_transform_term)
        normterm = tf.complex(
            tf.math.sqrt(
                tf.cast(
                    1 / input_pixel_size_m["x"] / input_pixel_number["x"] / input_pixel_size_m["y"] / input_pixel_number["y"] / output_pixel_size_m["x"] / output_pixel_size_m["y"],
                    dtype=dtype,
                )
            ),
            TF_ZERO,
        )
        ang_fx = tf.range(0, 1 / 2 / input_pixel_size_m["x"], 1 / input_pixel_size_m["x"] / input_pixel_number["x"], dtype=dtype)
        wavefront_outPlane = tf_generalSpline_regular1DGrid(kr / 2 / np.pi, ang_fx, wavefront_outPlane) * normterm
    else:
        normterm = tf.complex(
            tf.math.sqrt(
                tf.cast(
                    input_pixel_size_m["x"] * input_pixel_size_m["y"] / output_pixel_size_m["x"] / output_pixel_size_m["y"] / input_pixel_number["x"] / input_pixel_number["y"],
                    dtype=dtype,
                )
            ),
            TF_ZERO,
        )
        wavefront_outPlane = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(fourier_transform_term))) * normterm

    return tf.abs(wavefront_outPlane), tf.math.angle(wavefront_outPlane)


def fresnel_diffraction_coeffs(
    out_wavefront_ampl,
    out_wavefront_phase,
    wavelength_m,
    distance_m,
    output_pixel_size_m,
    output_pixel_number,
    dtype,
    radial_symmetry,
):
    """Adds the complex coefficient terms in the Fresnel diffraction integral formulation to the out-plane wavefront,
    computed by fresnel_diffraction_fft().

    Args:
        `out_wavefront_ampl` (tf.float): Field amplitude at the output plane, excluding complex coeffs, after propagating
            by the fresnel method, of shape (batch_size, output_pixel_number["x"], output_pixel_number["y"]) or
            (batch_size, 1, output_pixel_number["r"]).
        `out_wavefront_phase` (tf.float): Field phase at the output plane, excluding complex coeffs, after propagating
            by the fresnel method, of shape (batch_size, output_pixel_number["x"], output_pixel_number["y"]) or
            (batch_size, 1, output_pixel_number["r"]).
        `wavelength_m` (tf.float): tf constant corresponding to the wavelength of the field
            (should match that used in the fresnel_diffraction call)
        `distance_m` (tf.float64): tf constant corresponding to the distance propagated
            (should match that used in the fresnel_diffraction call)
        `output_pixel_size_m` (dict): Output field grid discretization/pitch in units of m, via dictionary {"x": float, "y": float}.
        `output_pixel_number` (dict): Output field grid length in terms of number of pixels, via dictionary {"x": float, "y": float}.
        `dtype` (tf.dtype): Datatype to be used in the tensorflow calculation; Only tf.float64 is currently supported.
        `radial_symmetry` (bool): Flag indicating if radial symmetry is used.

    Returns:
        `tf.float`: Field amplitude at the output plane with the complex coefficients added in. same shape as input.
        `tf.float`: Field phase at the output plane with the compelx coefficients added in. same shape as input.
    """

    TF_ZERO = tf.constant(0.0, dtype=dtype)

    # create the output plane coordinate grid
    output_pixel_x, output_pixel_y = tf_coordinate_grid(output_pixel_number, output_pixel_size_m, radial_symmetry, dtype)
    output_pixel_x = output_pixel_x[tf.newaxis]
    output_pixel_y = output_pixel_y[tf.newaxis]

    # add the final terms
    angular_wave_number = 2 * np.pi / wavelength_m
    quadterm = distance_m + (output_pixel_x**2 + output_pixel_y**2) / 2 / distance_m
    ### Neglect the power term, 1/i/lambda/z since we want psf normalized by energy at the end
    # wavefront = (
    #     tf.complex(wavefront_trans, TF_ZERO)
    #     * tf.complex(TF_ZERO, tf.cast(1.0 / wavelength_m / distance_m, dtype))
    #     * tf.exp(tf.complex(TF_ZERO, wavefront_phase + angular_wave_number * quadterm))
    # )
    wavefront = tf.complex(out_wavefront_ampl, TF_ZERO) * tf.exp(tf.complex(TF_ZERO, out_wavefront_phase + angular_wave_number * quadterm))

    return tf.math.abs(wavefront), tf.math.angle(wavefront)
