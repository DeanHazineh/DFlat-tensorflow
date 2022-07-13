import tensorflow as tf

from .psf_compute import psf_measured, wavefront_afterms_sensor
from .calc_ms_regularizer import regularize_ms_calc_tf
from .detectorResampling import sensorMeasurement_intensity_phase, sensorMeasurement_intensity_phase_radialData
from .hankel import radial_crop_or_pad, radial_conditional_resize_with_crop_or_pad


def broadband_batched_propagation(field_amplitude, field_phase, parameters_list):
    """Takes a batch of field amplitudes and field phases at an input plane and propagates the field to an output plane,
    for multiple wavelengths. 

    The input to output field distances is defined by parameters["sensor_distance_m"]. 
    The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    The output grid is defined via discretization of parameters["sensor_dx_m"] and number points of 
        parameters["sensor_pixel_number"]; these variable names are misnomyers as the architecture builds/reuses the
        functions initially written for psfs. 

    The set of input field amplitude and phases may be defined uniquely for each wavelength in wavelength_set_m or the
    input fields may be presumed to be the same across all the wavelengths. This function enables function overloading
    depending on the shape of the field_amplitude and phase tensor. 

    Args:
        `field_amplitude` (tf.float): Input plane field amplitude of shape 
            (len(wavelength_set_m), num_profiles, ms_samples["y"], ms_samples["x"]) or (len(wavelength_set_m), num_profiles, 1, ms_samples["r"]).
            Alternatively, the input field amplitude may be assumed the same across wavelengths and given via 
            (num_profiles, ms_samples["y"], ms_samples["x"]) or (num_profiles, 1, ms_samples["r"]).
        `field_phase` (tf.float): Input plane field phase of shape (len(wavelength_set_m), num_profiles, ms_samples["y"], ms_samples["x"])
            or (len(wavelength_set_m), num_profiles, 1, ms_samples["r"]). Alternatively, the input field phase may be 
            assumed the same across wavelengths and given via (num_profiles, ms_samples["y"], ms_samples["x"]) or
            (num_profiles, 1, ms_samples["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float`: Field amplitude of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        `tf.float`: Field phase of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
    """
    num_wavelengths = len(parameters_list)
    input_rank = tf.shape(field_amplitude).shape

    def lambda_loopCond(idx_, hold_ampl_, hold_phase_):
        return tf.less(idx_, num_wavelengths)

    if input_rank == 3:
        # the metasurface modulation profiles are the same across wavelength
        num_profiles = field_amplitude.shape[0]

        def lambda_loopBody(idx_, hold_ampl_, hold_phase_):
            ampl, phase = field_propagation(field_amplitude, field_phase, parameters_list[idx_])

            hold_ampl_ = tf.concat([hold_ampl_, tf.expand_dims(ampl, 0)], axis=0)
            hold_phase_ = tf.concat([hold_phase_, tf.expand_dims(phase, 0)], axis=0)

            idx_ += 1

            return [idx_, hold_ampl_, hold_phase_]

    elif input_rank == 4:
        # The metasurface modulations are defined for each wavelength channel
        num_profiles = field_amplitude.shape[1]

        def lambda_loopBody(idx_, hold_ampl_, hold_phase_):
            ampl, phase = field_propagation(field_amplitude[idx_], field_phase[idx_], parameters_list[idx_])

            hold_ampl_ = tf.concat([hold_ampl_, tf.expand_dims(ampl, 0)], axis=0)
            hold_phase_ = tf.concat([hold_phase_, tf.expand_dims(phase, 0)], axis=0)

            idx_ += 1

            return [idx_, hold_ampl_, hold_phase_]

    else:
        raise ValueError(
            "broadband_batched_psf_measured: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor."
        )

    # Run batched loop
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]
    radial_symmetry = parameters_list[0]["radial_symmetry"]
    num_pts_x = tf.cond(radial_symmetry, lambda: sensor_pixel_number["r"], lambda: sensor_pixel_number["x"])
    num_pts_y = tf.cond(radial_symmetry, lambda: 1, lambda: sensor_pixel_number["y"])
    dtype = parameters_list[0]["dtype"]

    idx = tf.constant(0, dtype=tf.int32)
    hold_ampl = tf.zeros((1, num_profiles, num_pts_y, num_pts_x), dtype=dtype)
    hold_phase = tf.zeros((1, num_profiles, num_pts_y, num_pts_x), dtype=dtype)

    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, hold_ampl, hold_phase])

    return (
        tf.stack(loopData[1][1:]),
        tf.stack(loopData[2][1:]),
    )


def field_propagation(field_amplitude, field_phase, parameters):
    """Takes a batch of field amplitudes and field phases at an input plane (of a single wavelength) and propagates the
    field to an output plane.

    The input to output field distances is defined by parameters["sensor_distance_m"]. 
    The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    The output grid is defined via discretization of parameters["sensor_dx_m"] and number points of 
    parameters["sensor_pixel_number"]; these variable names are used as the architecture builds/reuses the functions 
    initially written for computing psfs. 

    Args:
        `field_amplitude` (tf.float): Initial plane field amplitude, of shape 
            (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
        `field_phase` (tf.float): Initial plane field phase, of shape (batch_size, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float64`: Output plane field amplitude, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (1, batch_size, 1, sensor_pixel_number["r"]).
        `tf.float64`: Initial plane field phase, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (1, batch_size, 1, sensor_pixel_number["r"]). 
    """
    # Possibly add an assertions check here in the future if people are misusing.
    # .

    # For an accurate calculation, resample field and add the padding as defined during prop_param initialization
    field_amplitude, field_phase = regularize_ms_calc_tf(field_amplitude, field_phase, parameters)

    # Propagate the field, piggy-back off the psf derived functions
    field_amplitude, field_phase = wavefront_afterms_sensor(field_amplitude, field_phase, parameters)

    # # pad or crop to match the user defined number of pixels in the output
    # sensor_pixel_number = parameters["sensor_pixel_number"]
    # radial_symmetry = parameters["radial_symmetry"]
    # field_amplitude = radial_conditional_resize_with_crop_or_pad(field_amplitude, radial_symmetry, sensor_pixel_number)
    # field_phase = radial_conditional_resize_with_crop_or_pad(field_phase, radial_symmetry, sensor_pixel_number)

    # Reinterpolate to the user specified grid and also ensure resize
    if parameters["radial_symmetry"]:
        field_amplitude, field_phase = sensorMeasurement_intensity_phase_radialData(
            field_amplitude, field_phase, parameters
        )
    else:
        field_amplitude, field_phase = sensorMeasurement_intensity_phase(field_amplitude, field_phase, parameters)

    return field_amplitude, field_phase


def broadband_batched_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters_list):
    """Batch computes the PSF measured on the photosensor for metasurface modulation profiles on each wavelength channel
    within a set.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across 
    wavelength (as is true for real metasurfaces). If a rank 3 tensors, the set of modulation profiles are assumed to 
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case 
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent). 
    The output structure is the same but this function is overloaded on input.

    Args:
        `ms_trans` (tf.float): Metasurface transmittance profiles on each wavelength channel, of shape 
            (len(wavelength_set_m), num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or 
            (len(wavelength_set_m), num_profiles, 1, ms_samplesM["r"]). Alternatively, the transmittance of the optical
            element may be assumed wavelength-independent and given via shapes 
            (num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or 
            (num_profiles, 1, ms_samplesM["r"]).
        `ms_phase` (tf.float):  Metasurface phase profiles on each wavelength channel. The shape is the same as ms_trans input.
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
            (ideally to make the total psf energy unity).
        `point_source_locs` (tf.float): Tensor of point-source coordinates, of shape (N,3)
        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Batched PSF intensity of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: Batched PSF phase of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """
    # unpack parameters
    num_wavelengths = len(parameters_list)
    input_rank = tf.shape(ms_trans).shape

    def lambda_loopCond(idx_, holdPSF_int_, holdPSF_phase_):
        return tf.less(idx_, num_wavelengths)

    if input_rank == 3:
        # the metasurface modulation profiles are the same across wavelength
        num_ms = ms_trans.shape[0]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
            psfs_int, psfs_phase = batched_psf_measured(
                ms_trans, ms_phase, normby, point_source_locs, parameters_list[idx_]
            )

            holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=0)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=0)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_]

    elif input_rank == 4:
        # The metasurface modulations are defined for each wavelength channel
        num_ms = ms_trans.shape[1]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
            psfs_int, psfs_phase = batched_psf_measured(
                ms_trans[idx_], ms_phase[idx_], normby, point_source_locs, parameters_list[idx_]
            )

            holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=0)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=0)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_]

    else:
        raise ValueError(
            "broadband_batched_psf_measured: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor."
        )

    # Run batched loop with manual conditional overloading on function
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]
    dtype = parameters_list[0]["dtype"]
    num_ps = point_source_locs.shape[0]
    idx = tf.constant(0, dtype=tf.int32)

    holdPSF_int = tf.zeros((1, num_ms, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    holdPSF_phase = tf.zeros((1, num_ms, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdPSF_int, holdPSF_phase])

    return (
        tf.stack(loopData[1][1:]),
        tf.stack(loopData[2][1:]),
    )


def batched_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters):
    """Given a stack of metasurface transmission and phase profiles, compute the PSF for a set of point-sources.
    
    This function batches the psf_measured function call.
    
    Args:
        `ms_trans` (tf.float): Metasurface transmittance, of shape 
            (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"]).
        `ms_phase` (tf.float): Metasurface phase, the same shape as ms_trans.
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to divide the computed psf by,
            (ideally to make the total psf energy unity).
        `point_source_locs` (tf.float): Tensor of point-source coordinates, of shape (N,3)
        `parameters` (prop_param):  Settings object defining field propagation details.

    Returns:
        `tf.float`: Batched PSFs intensity of shape (1, batch_size, N, sensor_pixel_number["y"], sensor_pixel_number["x"]).
        `tf.float`: Batched PSFs phase of shape (1, batch_size, N, sensor_pixel_number["y"], sensor_pixel_number["x"]).
    """
    # unpack parameters
    num_ms = ms_trans.shape[0]

    # Use a tf while_loop to batch metasurfaces
    def ms_loopCond(idx_, holdPSF_int_, holdPSF_phase_):
        return tf.less(idx_, num_ms)

    def ms_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
        psfs_int, psfs_phase = psf_measured(
            point_source_locs, ms_trans[idx_ : idx_ + 1], ms_phase[idx_ : idx_ + 1], parameters, normby,
        )

        holdPSF_int_ = tf.concat([holdPSF_int_, tf.expand_dims(psfs_int, 0)], axis=0)
        holdPSF_phase_ = tf.concat([holdPSF_phase_, tf.expand_dims(psfs_phase, 0)], axis=0)

        idx_ += 1
        return [idx_, holdPSF_int_, holdPSF_phase_]

    # Run batched loop
    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]
    num_ps = point_source_locs.shape[0]
    idx = tf.constant(0, dtype=tf.int32)
    holdPSF_int = tf.zeros((1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    holdPSF_phase = tf.zeros((1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    loopData = tf.while_loop(ms_loopCond, ms_loopBody, loop_vars=[idx, holdPSF_int, holdPSF_phase])

    return (tf.expand_dims(tf.stack(loopData[1][1:]), 0), tf.expand_dims(tf.stack(loopData[2][1:]), 0))

