import tensorflow as tf
from .hankel import qdht, radial_2d_transform, radial_2d_transform_wrapped_phase, radial_2d_transform_complex
from .psf_compute import psf_sensor_LTI


def otf_sensor(
    point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters,
):
    """Computes the optical transfer function using the linear, translational invariant assumptions for the optical system.
    The OTF characterizes the optical metasurface and the geometry. This function does not take into account the finite 
    pixel size.

    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (N,3) to compute the PSF for 
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (1, ms_samplesM['y'], ms_samplesM['x'])
            or (1, 1, ms_samplesM["r]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (1, ms_samplesM['y'], ms_samplesM['x'])
            or (1, 1, ms_samplesM["r]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float64`: Center-normalized OTF amplitude 
        `tf.float64`: OTF phase 
    """

    radial_symmetry = parameters["radial_symmetry"]

    # First get the coherent LTI PSF on the sensor grid
    # OTF is defined by the psf intensity and does not require phase
    psf_intensity, _ = psf_sensor_LTI(
        point_source_locs,
        ms_modulation_trans,
        ms_modulation_phase,
        parameters,
        tf.constant(1.0, dtype=tf.float64),
        addCoeffs=False,
    )
    psf_intensity = tf.complex(psf_intensity, tf.cast(0.0, dtype=tf.float64))

    # Take the fourier transform using Hankel or 2D-fft
    if radial_symmetry:
        calc_samplesN = parameters["calc_samplesN"]
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
        dtype = parameters["dtype"]
        sensor_r = tf.range(0, calc_samplesN["r"], dtype=dtype) * calc_sensor_dx_m["x"]
        kr, OTF = qdht(sensor_r, psf_intensity)
    else:
        OTF = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(psf_intensity)))

    # Apply the standard OTF center normalization convention
    normby = tf.complex(tf.math.reduce_max(tf.math.abs(OTF), axis=[1, 2]), tf.cast(0.0, dtype=tf.float64))
    OTF = OTF / tf.expand_dims(tf.expand_dims(normby, -1), -1)

    # After calculation is done, if radial symmetry was used, convert back to 2D
    if radial_symmetry:
        OTF_ampl = radial_2d_transform(tf.squeeze(tf.math.abs(OTF), 1))
        OTF_phase = radial_2d_transform_wrapped_phase(tf.squeeze(tf.math.angle(OTF), 1))
    else:
        OTF_ampl = tf.math.abs(OTF)
        OTF_phase = tf.math.angle(OTF)

    return OTF_ampl, OTF_phase


def atf_sensor(
    point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters,
):
    """Computes the amplitude transfer function using the linear, translational invariant assumptions for the optical 
    system. The ATF characterizes the optical metasurface and the geometry. This function does not take into account 
    the finite pixel size.

    Args:
        `point_source_locs` (tf.float64): Set of point-source coordinates of shape (N,3) to compute the PSF for 
        `ms_modulation_trans` (tf.float64): Metasurface transmittance profile of shape (1, ms_samplesM['y'], ms_samplesM['x']) 
            or (1, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float64): Metasurface phase profile of shape (1, ms_samplesM['y'], ms_samplesM['x'])
            or (1, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float64`: ATF amplitude 
        `tf.float64`: ATF phase 
    
    """

    # Get the ATF by fourier transforming the PSF
    # Note this is not the same as just the pupil function because of the quadratic phase terms added
    # after the doing the fresnel transform
    dtype = parameters["dtype"]
    TF_ZERO = tf.cast(0.0, dtype=dtype)
    radial_symmetry = parameters["radial_symmetry"]

    psf_intensity, psf_phase = psf_sensor_LTI(
        point_source_locs,
        ms_modulation_trans,
        ms_modulation_phase,
        parameters,
        tf.constant(1.0, dtype=dtype),
        addCoeffs=True,
    )
    psf = tf.complex(tf.math.sqrt(psf_intensity, TF_ZERO)) * tf.exp(tf.complex(TF_ZERO, psf_phase))

    # Fourier transform of the complex PSF
    if radial_symmetry:
        calc_samplesN = parameters["calc_samplesN"]
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
        sensor_r = tf.range(0, calc_samplesN["r"], dtype=dtype) * calc_sensor_dx_m["x"]
        kr, ATF = qdht(sensor_r, psf)
    else:
        ATF = tf.signal.fftshift(tf.signal.fft2d(tf.signal.ifftshift(psf)))

    # After calculation is done, if radial symmetry was used, convert back to 2D
    if radial_symmetry:
        ATF = radial_2d_transform_complex(tf.squeeze(ATF, 1))

    return tf.math.abs(ATF), tf.math.angle(ATF)

