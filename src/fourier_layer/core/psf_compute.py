import tensorflow as tf
import numpy as np
from .calc_ms_regularizer import regularize_ms_calc_tf
from .hankel import radial_2d_transform, radial_2d_transform_wrapped_phase
from .fresnel_integral_method import fresnel_diffraction_coeffs, fresnel_diffraction_fft
from .angular_spectrum_method import transfer_function_diffraction
from .detectorResampling import sensorMeasurement_intensity_phase


def psf_measured(
    point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters, normby_transmittance,
):
    """Computes the point-spread function at the sensor-plane then resamples and integrates to yield measurement on a 
    user-specified detector pixels. 

    Generally, one should set the sensor plane grid to be finer than the detector pixel grid. Area integration is then 
    used over the detector pixel area for measured intensity while area averaging is used for measured phase.

    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (N,3) to compute the PSF for 
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (1, ms_samplesM['y'], ms_samplesM['x'])
            or (1, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (1, ms_samplesM['y'], ms_samplesM['x']) or
            (1, 1, ms_samplesM["r"])
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.

    Returns:
        `tf.float`: Field intensity measured on the detector, of shape (N, sensor_pixel_number["y"], sensor_pixel_number["x"]).
        `tf.float`: Fied phase measured on the detector, of shape (N, sensor_pixel_number["y"], sensor_pixel_number["x"]).
    """

    # compute the PSF at the sensor plane -- note that psf_sensor returns
    # tf.math.abs(field)**2 already with appropriate psf normalization on energy!
    calc_modulation_intensity, calc_modulation_phase = psf_sensor(
        point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters, normby_transmittance
    )

    # Predict the measurement on specified detector pixel size and shape
    (calc_modulation_intensity, calc_modulation_phase,) = sensorMeasurement_intensity_phase(
        calc_modulation_intensity, calc_modulation_phase, parameters
    )

    return calc_modulation_intensity, calc_modulation_phase


def psf_sensor(
    point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters, normby_transmittance, convert_2D=True
):
    """ Computes the point-spread function on a unifrom grid at the sensor-plane, given a metasurface phase and transmittance.
    
    If in the paraxial regime, fresnel engine is used, and if point_source_locs are on the optical axis, this calculation is the same 
    as psf_sensor_LTI. When off-axis point-sources are specified, this function differs.
    
    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (N,3) to compute the PSF for 
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (1, ms_samplesM['y'], ms_samplesM['x'])
            or (1, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (1, ms_samplesM['y'], ms_samplesM['x']) 
            or (1, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field.

    Returns:
        `tf.float`: Field intensity at the sensor-plane grid of shape (N, calc_ms_samplesM['y'], calc_ms_samplesM['x'])
        `tf.float`: Field phase at the sensor-plane grid of shape (N, calc_ms_samplesM['y'], calc_ms_samplesM['x']).
    """

    # Run psf calculation with assertions upheld
    all_assertions = psf_sensor_assertions(point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters)
    with tf.control_dependencies(all_assertions):

        # For an accurate calculation, resample ms and add the padding as defined during prop_param initialization
        calc_modulation_trans, calc_modulation_phase = regularize_ms_calc_tf(
            ms_modulation_trans, ms_modulation_phase, parameters
        )

        # Get the field after the metasurface, given a point-source spherical wave origin
        calc_modulation_trans, calc_modulation_phase = wavefront_pointSources_afterms(
            point_source_locs, calc_modulation_trans, calc_modulation_phase, parameters
        )

        # get finely sampled field just above the sensor (radial converted to 2D psf at end)
        (calc_modulation_trans, calc_modulation_phase,) = wavefront_afterms_sensor(
            calc_modulation_trans, calc_modulation_phase, parameters,
        )

        # After calculation is done, if radial symmetry was used, convert back to 2D unless override return radial
        if parameters["radial_symmetry"] and convert_2D:
            calc_modulation_trans = radial_2d_transform(tf.squeeze(calc_modulation_trans, 1))
            calc_modulation_phase = radial_2d_transform_wrapped_phase(tf.squeeze(calc_modulation_phase, 1))

        ### Normalize by input source energy factor
        calc_modulation_trans /= normby_transmittance
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]

    return (
        tf.math.abs(calc_modulation_trans) ** 2 * calc_sensor_dx_m["y"] * calc_sensor_dx_m["x"],
        calc_modulation_phase,
    )


def wavefront_afterms_sensor(
    calc_modulation_trans, calc_modulation_phase, parameters,
):
    """Propagate the complex field from after the ms to just above the sensor plane

    Args:
        `calc_modulation_trans` (tf.float): Field amplitude just after the metasurface (upsampled/padded), of shape 
            (N, calc_samplesN['y'], calc_samplesN['x']) or (N, 1, calc_samplesN["r"]).
        `calc_modulation_phase` (tf.float): Field phase just after the metasurface (upsampled/padded), of shape 
            (N, calc_samplesN['y'], calc_samplesN['x']) or (N, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Field amplitude at the sensor plane grid, of shape (N, calc_samplesN['y'], calc_samplesN['x']) or (N, 1, calc_samplesN["r"]).
        `tf.float64`: Field phase at the sensor plane grid, of shape (N, calc_samplesN['y'], calc_samplesN['x']) or (N, 1, calc_samplesN["r"]).
    """

    # propagate the field using a specified engine
    wavelength_m = parameters["wavelength_m"]
    sensor_distance_m = parameters["sensor_distance_m"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    dtype = parameters["dtype"]
    radial_symmetry = parameters["radial_symmetry"]
    diffractionEngine = parameters["diffractionEngine"]

    if diffractionEngine == "fresnel_fourier":
        propagator = fresnel_diffraction_fft
    elif diffractionEngine == "ASM_fourier":
        propagator = transfer_function_diffraction

    wavefront_trans, wavefront_phase = propagator(
        calc_modulation_trans,
        calc_modulation_phase,
        wavelength_m,
        sensor_distance_m,
        calc_ms_dx_m,
        calc_samplesN,
        calc_sensor_dx_m,
        dtype,
        radial_symmetry,
    )

    # When the fresnel transform calculation is done, coefficients need to be added back in
    # this is done here rather than in the propagator call so that all propagator engines have same
    # inputs to function. No Coefficients are missing in the transfer_function_diffraction propagator
    if diffractionEngine == "fresnel_fourier":
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
        wavefront_trans, wavefront_phase = fresnel_diffraction_coeffs(
            wavefront_trans,
            wavefront_phase,
            wavelength_m,
            sensor_distance_m,
            calc_sensor_dx_m,
            calc_samplesN,
            dtype,
            radial_symmetry,
        )

    return wavefront_trans, wavefront_phase


def wavefront_pointSources_afterms(
    point_sources_locs, calc_modulation_trans, calc_modulation_phase, parameters,
):
    """Computes the set of complex fields after a metasurface, resulting from the illuminated, upsampled/padded phase
    and transmittance modulation profiles. The incident wavefront at the metasurface corresponds to spherical wavefronts
    originating at the point-source locations. 

    Args:
        `point_sources_locs` (tf.float): Set of point-source coordinates to compute PSF for, of shape (N,3).
        `calc_modulation_trans` (tf.float): Metasurface transmittance (upsampled/padded) of shape (1, calc_samplesN['y'], calc_samplesN['x'])
            or (1, 1, calc_samplesN["r"]).
        `calc_modulation_phase` (tf.float): Metasurface phase (upsampled/padded) of shape (1, calc_samplesN['y'], calc_samplesN['x'])
            or (1, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float`: Field amplitude after the metasurface, the same shape as the input.
        `tf.float`: Field phase after the metasurface, the same shape as the input.
    """

    # unpack the parameters
    wavelength_m = parameters["wavelength_m"]
    radial_symmetry = parameters["radial_symmetry"]
    dtype = parameters["dtype"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    angular_wave_number = 2 * np.pi / wavelength_m
    TF_ZERO = tf.constant(0.0, dtype=tf.float64)

    # create the metasurface grid
    if radial_symmetry:
        calc_pixel_x, calc_pixel_y = tf.meshgrid(tf.range(calc_samplesN["r"], dtype=dtype), tf.range(1, dtype=dtype),)
    else:
        calc_pixel_x, calc_pixel_y = tf.meshgrid(
            tf.range(calc_samplesN["x"], dtype=dtype), tf.range(calc_samplesN["y"], dtype=dtype),
        )
        calc_pixel_x = calc_pixel_x - (calc_pixel_x.shape[1] - 1) / 2
        calc_pixel_y = calc_pixel_y - (calc_pixel_y.shape[0] - 1) / 2
    calc_pixel_x = tf.expand_dims(calc_pixel_x * calc_ms_dx_m["x"], 0)
    calc_pixel_y = tf.expand_dims(calc_pixel_y * calc_ms_dx_m["y"], 0)

    # index of the points
    point_source_loc_x = point_sources_locs[:, 0]
    point_source_loc_y = point_sources_locs[:, 1]
    point_source_loc_z = point_sources_locs[:, 2]

    # computation of distance
    point_source_loc_x = tf.expand_dims(tf.expand_dims(point_source_loc_x, -1), -1)
    point_source_loc_y = tf.expand_dims(tf.expand_dims(point_source_loc_y, -1), -1)
    point_source_loc_z = tf.expand_dims(tf.expand_dims(point_source_loc_z, -1), -1)
    distance_point_ms = tf.sqrt(
        (calc_pixel_x - point_source_loc_x) ** 2 + (calc_pixel_y - point_source_loc_y) ** 2 + point_source_loc_z ** 2
    )

    ## Compute product of spherical wavefront and metasurface
    ## For the spherical wave, one could place 1/(i*lambda*r) instead of 1/r
    ## since intensity and energy in fact requires the extra term as in

    # wavefront = tf.complex(calc_modulation_trans / distance_point_ms, TF_ZERO) * tf.exp(
    #     tf.complex(TF_ZERO, calc_modulation_phase + angular_wave_number * distance_point_ms)
    # )

    ## However we remove the 1/r and 1/lambda dependence to aid in normalized psf downstream
    wavefront = tf.complex(calc_modulation_trans, TF_ZERO) * tf.exp(
        tf.complex(TF_ZERO, calc_modulation_phase + angular_wave_number * distance_point_ms)
    )

    return tf.math.abs(wavefront), tf.math.angle(wavefront)


def psf_sensor_assertions(point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters):
    # create assertions to be controlled when running psf_sensor
    dtype = parameters["dtype"]
    TF_ZERO = tf.constant(0.0, dtype=dtype)

    # assert1 = tf.debugging.assert_equal(
    #     tf.sort(point_source_locs[:, 2]),
    #     point_source_locs[:, 2],
    #     name="point_source_sequence_assertion",
    #     summarize="Point locations must be sorted from close to far",
    # )

    assert2 = tf.debugging.assert_equal(
        point_source_locs.shape[1],
        3,
        name="point_source_shape_assertion",
        summarize="Point source locations should be Nx3 in dimension",
    )

    assert3 = tf.debugging.assert_greater(
        point_source_locs[:, 2],
        TF_ZERO,
        name="point_source_positive_z_assertion",
        summarize="Point source locations should have positive z values",
    )

    assert4 = tf.debugging.Assert(
        dtype == tf.float64, dtype, name="datatype_assertion", summarize="calculation data type must be tf.float64",
    )

    # Check the shape of the data to make sure no mistakes were made
    # in writing new functions
    assert5 = tf.debugging.assert_equal(
        tf.shape(ms_modulation_trans).shape,
        3,
        name="lens_intensity_dimension_assertion",
        summarize="Dimensionality of lens transmittance must be 3",
    )

    assert6 = tf.debugging.assert_equal(
        tf.shape(ms_modulation_phase).shape,
        3,
        name="lens_phase_dimension_assertion",
        summarize="Dimension of lens phase must be 3",
    )

    # check that the input dimensions match what is expected by parameters
    radial_symmetry = parameters["radial_symmetry"]
    ms_samplesM = parameters["ms_samplesM"]
    ms_dimension = tf.cond(
        radial_symmetry,
        lambda: tf.cast([1, ms_samplesM["r"]], tf.int32),
        lambda: tf.cast([ms_samplesM["y"], ms_samplesM["x"]], tf.int32),
    )

    assert7 = tf.debugging.assert_equal(
        ms_dimension,
        tf.shape(ms_modulation_trans)[1:],
        name="ms_intensity_dimension_assertion",
        summarize="Check ms transmittance shape",
    )

    assert8 = tf.debugging.assert_equal(
        ms_dimension,
        tf.shape(ms_modulation_phase)[1:],
        name="ms_phase_dimension_assertion",
        summarize="Check ms phase shape",
    )

    assert9 = tf.debugging.assert_type(point_source_locs, tf.float64, name="point_source_locs_dtype_assetion")

    assert10 = tf.debugging.assert_type(ms_modulation_trans, tf.float64, name="ms_intensity_dtype_assertion")

    assert11 = tf.debugging.assert_type(ms_modulation_phase, tf.float64, name="ms_phase_dtype_assertion")

    # return assertions as list
    all_assertions = [
        # assert1,
        assert2,
        assert3,
        assert4,
        assert5,
        assert6,
        assert7,
        assert8,
        assert9,
        assert10,
        assert11,
    ]

    return all_assertions


def psf_sensor_LTI(
    point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters, normby_transmittance, addCoeffs=True
):
    """Computes the point-spread function on a uniform grid at the sensor plane, given a metasurface and phase transmittance.

    This calculation is formulated in a way that is most consistent to the linear, translationally invariant treatment 
    used for approximate rendering. See technical documentation for more details and how it differs from psf_sensor. 
    This is only defined for the fresnel diffraction integral propagation treatment and inherrently assumes paraxial optics.

    Args:
        `point_source_locs` (tf.float): Set of point-source coordinates of shape (N,3) to compute the PSF for 
        `ms_modulation_trans` (tf.float): Metasurface transmittance profile of shape (1, ms_samplesM['y'], ms_samplesM['x']) 
            or (1, 1, ms_samplesM["r"]).
        `ms_modulation_phase` (tf.float): Metasurface phase profile of shape (1, ms_samplesM['y'], ms_samplesM['x']) or 
            (1, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.
        `normby_transmittance` (tf.float): Scalar normalization factor for PSF field of shape (N,1).
        `addCoeffs` (bool, optional): Boolean flag to include the correct phase coefficients.

    Returns:
        `tf.float64`: Intensity at the sensorplane of the LTI PSF, of shape (N, calc_ms_samplesM['y'], calc_ms_samplesM['x']).
        `tf.float64`: Phase at the sensorplane of the LTI PSF, of shape (N, calc_ms_samplesM['y'], calc_ms_samplesM['x']).
    """

    # Get list of assertions used in the regular PSF calculation and add fresnel assertion
    all_assertions = psf_sensor_assertions(point_source_locs, ms_modulation_trans, ms_modulation_phase, parameters)
    assertFresnel = tf.debugging.assert_equal(
        parameters["diffractionEngine"],
        "fresnel_fourier",
        name="paraxial_psf_expectation_check",
        summarize="Ensure paraxial psf is expected when calling for LTI psf",
    )
    all_assertions.append(assertFresnel)
    with tf.control_dependencies(all_assertions):

        # For an accurate calculation, resample ms and add the padding as defined during param initialization
        calc_modulation_trans, calc_modulation_phase = regularize_ms_calc_tf(
            ms_modulation_trans, ms_modulation_phase, parameters
        )
        print(calc_modulation_trans.shape)

        # Get finely sampled field just above the sensor
        (calc_modulation_trans, calc_modulation_phase,) = wavefront_pointsources_sensor_LTI(
            point_source_locs, calc_modulation_trans, calc_modulation_phase, parameters, addCoeffs
        )
        print(calc_modulation_trans.shape)

        # After calculation is done, if radial symmetry was used, convert back to 2D
        if parameters["radial_symmetry"]:
            calc_modulation_trans = radial_2d_transform(tf.squeeze(calc_modulation_trans, 1))
            calc_modulation_phase = radial_2d_transform_wrapped_phase(tf.squeeze(calc_modulation_phase, 1))

        ### Normalize by input source energy factor
        calc_modulation_trans /= normby_transmittance
        calc_sensor_dx_m = parameters["calc_sensor_dx_m"]

    return (
        tf.math.abs(calc_modulation_trans) ** 2 * calc_sensor_dx_m["y"] * calc_sensor_dx_m["x"],
        calc_modulation_phase,
    )


def wavefront_pointsources_sensor_LTI(
    point_sources_locs, calc_modulation_trans, calc_modulation_phase, parameters, addCoeffs=True
):
    """Computes the set of complex fields at the sensor grid, resulting from the illuminated, upsampled/padded phase 
    and transmittance modulation profiles. The incident wavefront at the metasurface corresponds to quadratic wavefronts 
    originating on axis and at depths specified in point_sources_locs. This function applies a linear, transtlationally
     invariant formulation. If addCoeffs is true, the appropriate LTI phase terms are included.
     
    Args:
        `point_sources_locs` (tf.float): Set of point-source coordinates to compute PSF for, of shape (N,3).
        `calc_modulation_trans` (tf.float): Metasurface transmittance (upsampled/padded) of shape (1, calc_samplesN['y'], calc_samplesN['x']) 
            or (1, 1, calc_samplesN["x"]).
        `calc_modulation_phase` (tf.float): Metasurface phase (upsampled/padded) of shape (1, calc_samplesN['y'], calc_samplesN['x'])
            or (1, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.
        `addCoeffs` (bool, optional): Boolean flag to include the correct phase coefficients.

    Returns:
        `tf.float64`: Field amplitude after the metasurface, same shape as input
        `tf.float64`: Field phase after the metasurface, same shape as input
    """

    # unpack the parameters
    wavelength_m = parameters["wavelength_m"]
    radial_symmetry = parameters["radial_symmetry"]
    dtype = parameters["dtype"]
    calc_samplesN = parameters["calc_samplesN"]
    calc_ms_dx_m = parameters["calc_ms_dx_m"]
    sensor_distance_m = parameters["sensor_distance_m"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]

    # create the metasurface plane grid
    if radial_symmetry:
        calc_pixel_x, calc_pixel_y = tf.meshgrid(tf.range(calc_samplesN["r"], dtype=dtype), tf.range(1, dtype=dtype),)
    else:
        calc_pixel_x, calc_pixel_y = tf.meshgrid(
            tf.range(calc_samplesN["x"], dtype=dtype), tf.range(calc_samplesN["y"], dtype=dtype),
        )
        calc_pixel_x = calc_pixel_x - (calc_pixel_x.shape[1] - 1) / 2
        calc_pixel_y = calc_pixel_y - (calc_pixel_y.shape[0] - 1) / 2
    calc_pixel_x = tf.expand_dims(calc_pixel_x * calc_ms_dx_m["x"], 0)
    calc_pixel_y = tf.expand_dims(calc_pixel_y * calc_ms_dx_m["y"], 0)

    # get z distance of pointsources and apply quadratic wavefronts
    point_source_loc_z = point_sources_locs[:, 2]
    point_source_loc_z = tf.expand_dims(tf.expand_dims(point_source_loc_z, -1), -1)
    angular_wave_number = 2 * np.pi / wavelength_m
    quadWavefront = (calc_pixel_x ** 2 + calc_pixel_y ** 2) / 2 / point_source_loc_z + point_source_loc_z

    TF_ZERO = tf.cast(0.0, dtype=dtype)
    ## The quadratic wavefront has factor 1/z but the true factor is 1/(i*wavelength*z)
    # wavefront_afterlens = tf.complex(calc_modulation_trans / point_source_loc_z, TF_ZERO) * tf.exp(
    #     tf.complex(TF_ZERO, calc_modulation_phase + angular_wave_number * quadWavefront)
    # )
    ## As done in wavefront_pointSources_afterms, we neglect the 1/z to aid normalization of psf to energy
    wavefront_afterlens = tf.complex(calc_modulation_trans, TF_ZERO) * tf.exp(
        tf.complex(TF_ZERO, calc_modulation_phase + angular_wave_number * quadWavefront)
    )

    # Propagate the field to the sensor using the fresnel method
    wavefront_ampl, wavefront_phase = fresnel_diffraction_fft(
        tf.math.abs(wavefront_afterlens),
        tf.math.angle(wavefront_afterlens),
        wavelength_m,
        sensor_distance_m,
        calc_ms_dx_m,
        calc_samplesN,
        calc_sensor_dx_m,
        dtype,
        radial_symmetry,
    )

    if addCoeffs:
        # Add the fresnel coefficients back in
        wavefront_ampl, wavefront_phase = fresnel_diffraction_coeffs(
            wavefront_ampl,
            wavefront_phase,
            wavelength_m,
            sensor_distance_m,
            calc_sensor_dx_m,
            calc_samplesN,
            dtype,
            radial_symmetry,
        )

        # Add the LTI coefficient (This is the replacement of the object space quad wavefront with a magnified object space quad wavefront)
        # as explained in goodman ed 4, ch 6.3.2
        wavefront_phase += LTI_sensorplane_phaseterm(
            point_sources_locs,
            wavelength_m,
            sensor_distance_m,
            calc_sensor_dx_m,
            calc_samplesN,
            dtype,
            radial_symmetry,
        )

    return wavefront_ampl, wavefront_phase


def LTI_sensorplane_phaseterm(
    point_sources_locs, wavelength_m, distance_m, output_pixel_size_m, output_pixel_number, dtype, radial_symmetry,
):
    """Returns the approximate phase terms for the LTI formulation at the output grid. 

    Args:
        `point_sources_locs` (tf.float): Set of point-source coordinates to compute PSF for, of shape (N,3).
        `wavelength_m` (tf.float): Tf constant corresponding to the wavelength of the field (should match that used in the fresnel_diffraction call)
        `distance_m` (tf.float): Tf constant corresponding to the distance propagated (should match that used in the fresnel_diffraction call)
        `output_pixel_size_m` (dict): Output field grid discretization/pitch in units of m, via dictionary {"x": float, "y": float}.
        `output_pixel_number` (dict): Output field grid length in terms of number of pixels, via dictionary {"x": float, "y": float}.
        `dtype` (tf.dtype): Datatype to be used in the tensorflow calculation; Only tf.float64 is currently supported.
        `radial_symmetry` (bool): Flag indicating if radial symmetry is used. 
        
    Returns
        `tf.float64`: Phase term to be added back to a wavefront phase at the output plane, of shape (N,  calc_samplesN["y"], calc_samplesN["x"])
            or (1, 1, calc_samplesN["r"]).
    """

    # Define sensor plane coords
    if radial_symmetry:
        output_pixel_x, output_pixel_y = tf.meshgrid(
            tf.range(output_pixel_number["r"], dtype=dtype), tf.range(1, dtype=dtype),
        )
    else:
        output_pixel_x, output_pixel_y = tf.meshgrid(
            tf.range(output_pixel_number["x"], dtype=dtype), tf.range(output_pixel_number["y"], dtype=dtype),
        )
        output_pixel_x = output_pixel_x - (tf.shape(output_pixel_x)[1] - 1) / 2
        output_pixel_y = output_pixel_y - (tf.shape(output_pixel_y)[0] - 1) / 2
    output_pixel_x = output_pixel_x * output_pixel_size_m["x"]
    output_pixel_y = output_pixel_y * output_pixel_size_m["y"]
    output_pixel_x = tf.expand_dims(output_pixel_x, 0)
    output_pixel_y = tf.expand_dims(output_pixel_y, 0)

    # Create the magnification, dependent quad phase factor
    point_source_loc_z = point_sources_locs[:, 2]
    point_source_loc_z = tf.expand_dims(tf.expand_dims(point_source_loc_z, -1), -1)
    angular_wave_number = 2 * np.pi / wavelength_m

    magnification = -distance_m / point_source_loc_z
    phaseterm = (
        angular_wave_number / 2 / point_source_loc_z / magnification ** 2 * (output_pixel_x ** 2 + output_pixel_y ** 2)
    )

    return phaseterm
