import tensorflow as tf

from .compute_psf import (
    psf_measured,
    psf_measured_MatrixASM,
    field_propagation,
    field_propagation_MatrixASM,
)

### Propagation Layer Routines
def loopBatch_field_propagation(field_amplitude, field_phase, parameters):
    """Loops the field propagation routine over batch. This was written for the Propagate_Planes_Layer_Mono, i.e. monochromatic.

    Args:
        `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
            (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
        `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
            (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
        `parameters` (prop_params): Single prop_params object with wavelength_m defined

    Returns:
        `tf.float`: Output plane field amplitude, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (1, batch_size, 1, sensor_pixel_number["r"]).
        `tf.float`: Initial plane field phase, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (1, batch_size, 1, sensor_pixel_number["r"]).
    """

    # unpack parameters
    num_ms = field_amplitude.shape[0]

    # define loop condition
    def lambda_loopCond(idx_, holdField_ampl_, holdField_phase_):
        return tf.less(idx_, num_ms)

    # The loop body
    def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_):
        fields_ampl, fields_phase = field_propagation(field_amplitude[idx_ : idx_ + 1], field_phase[idx_ : idx_ + 1], parameters)
        holdField_ampl_ = tf.concat([holdField_ampl_, fields_ampl], axis=0)
        holdField_phase_ = tf.concat([holdField_phase_, fields_phase], axis=0)

        idx_ += 1

        return [idx_, holdField_ampl_, holdField_phase_]

    sensor_pixel_number = parameters["sensor_pixel_number"]
    radial_symmetry = parameters["radial_symmetry"]
    num_pts_x = tf.cond(radial_symmetry, lambda: sensor_pixel_number["r"], lambda: sensor_pixel_number["x"])
    num_pts_y = tf.cond(radial_symmetry, lambda: 1, lambda: sensor_pixel_number["y"])
    dtype = parameters["dtype"]
    idx = tf.constant(0, dtype=tf.int32)

    holdField_ampl = tf.zeros((1, num_pts_y, num_pts_x), dtype=dtype)
    holdField_phase = tf.zeros((1, num_pts_y, num_pts_x), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdField_ampl, holdField_phase])

    return (
        tf.stack(loopData[1][1:, :, :]),
        tf.stack(loopData[2][1:, :, :]),
    )


def loopWavelength_field_propagation(field_amplitude, field_phase, parameters_list):
    """Loops the field propagation routine over wavelength. If one wants to compute over wavelength without batching, the
    X caller should be used instead which implements a matrix broadband calculation using the ASM engine.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (profile_batch, 1, ms_samplesM["r"]).

        `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).

        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Field amplitude of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        `tf.float`: Field phase of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"])."""

    # unpack parameters
    num_wavelengths = len(parameters_list)
    input_rank = tf.rank(field_amplitude)

    # define loop condition
    def lambda_loopCond(idx_, holdField_ampl_, holdField_phase_):
        return tf.less(idx_, num_wavelengths)

    # define loop body dependent on overloading input of rank 3 vs rank 4
    if input_rank == tf.TensorShape(3):
        # the metasurface modulation profiles are the same across wavelength
        num_ms = field_amplitude.shape[0]

        def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_):
            ampl, phase = field_propagation(field_amplitude, field_phase, parameters_list[idx_])
            holdField_ampl_ = tf.concat([holdField_ampl_, tf.expand_dims(ampl, 0)], axis=0)
            holdField_phase_ = tf.concat([holdField_phase_, tf.expand_dims(phase, 0)], axis=0)

            idx_ += 1

            return [idx_, holdField_ampl_, holdField_phase_]

    elif input_rank == tf.TensorShape(4):
        # The metasurface modulations are defined for each wavelength channel
        num_ms = field_amplitude.shape[1]

        def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_):
            ampl, phase = field_propagation(field_amplitude[idx_], field_phase[idx_], parameters_list[idx_])
            holdField_ampl_ = tf.concat([holdField_ampl_, tf.expand_dims(ampl, 0)], axis=0)
            holdField_phase_ = tf.concat([holdField_phase_, tf.expand_dims(phase, 0)], axis=0)

            idx_ += 1

            return [idx_, holdField_ampl_, holdField_phase_]

    else:
        raise ValueError("broadband field propagation: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    # Run batched loop with manual conditional overloading on function
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]
    radial_symmetry = parameters_list[0]["radial_symmetry"]
    num_pts_x = tf.cond(radial_symmetry, lambda: sensor_pixel_number["r"], lambda: sensor_pixel_number["x"])
    num_pts_y = tf.cond(radial_symmetry, lambda: 1, lambda: sensor_pixel_number["y"])
    dtype = parameters_list[0]["dtype"]
    idx = tf.constant(0, dtype=tf.int32)

    holdField_ampl = tf.zeros((1, num_ms, num_pts_y, num_pts_x), dtype=dtype)
    holdField_phase = tf.zeros((1, num_ms, num_pts_y, num_pts_x), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdField_ampl, holdField_phase])

    return (
        tf.stack(loopData[1][1:]),
        tf.stack(loopData[2][1:]),
    )


def batch_loopWavelength_field_propagation(field_amplitude, field_phase, parameters_list):
    """Loops the field propagation routine over wavelength and also adds a batch loop over the field input profiles dimension.
    To not explicitly batch over the stack of input fields, use loopWavelength_field_propagation instead. See there for more details.

    When the profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (profile_batch, 1, ms_samplesM["r"]).

        `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
            (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
            is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
            or (batch_size, 1, ms_samplesM["r"]).

        `parameters_list` (list): List of prop_param objects, each being initialized (in order of wavelength_set_m) for
            a different wavelength_m value.

    Returns:
        `tf.float`: Field amplitude of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        `tf.float`: Field phase of shape (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
    """

    # unpack parameters
    input_rank = tf.rank(field_amplitude)
    num_wavelengths = len(parameters_list)

    # define loop condition
    def lambda_loopCond(idx_, holdField_ampl_, holdField_phase_, num_ms_):
        return tf.less(idx_, num_ms_)

    # define loop body dependent on overloading input of rank 3 vs rank 4
    if input_rank == tf.TensorShape(3):
        # the metasurface modulation profiles are the same across wavelength
        num_ms = field_amplitude.shape[0]

        def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_, num_ms_):
            ampl, phase = loopWavelength_field_propagation(field_amplitude[idx_ : idx_ + 1], field_phase[idx_ : idx_ + 1], parameters_list)

            holdField_ampl_ = tf.concat([holdField_ampl_, ampl], axis=1)
            holdField_phase_ = tf.concat([holdField_phase_, phase], axis=1)

            idx_ += 1

            return [idx_, holdField_ampl_, holdField_phase_, num_ms_]

    elif input_rank == tf.TensorShape(4):
        # The metasurface modulations are defined for each wavelength channel
        num_ms = field_amplitude.shape[1]

        def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_, num_ms_):
            ampl, phase = loopWavelength_field_propagation(
                field_amplitude[:, idx_ : idx_ + 1, :, :], field_phase[:, idx_ : idx_ + 1, :, :], parameters_list
            )

            holdField_ampl_ = tf.concat([holdField_ampl_, ampl], axis=1)
            holdField_phase_ = tf.concat([holdField_phase_, phase], axis=1)

            idx_ += 1

            return [idx_, holdField_ampl_, holdField_phase_, num_ms_]

    else:
        raise ValueError("broadband field propagation: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    # Run batched loop with manual conditional overloading on function
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]
    radial_symmetry = parameters_list[0]["radial_symmetry"]
    num_pts_x = tf.cond(radial_symmetry, lambda: sensor_pixel_number["r"], lambda: sensor_pixel_number["x"])
    num_pts_y = tf.cond(radial_symmetry, lambda: 1, lambda: sensor_pixel_number["y"])
    dtype = parameters_list[0]["dtype"]
    idx = tf.constant(0, dtype=tf.int32)

    holdField_ampl = tf.zeros((num_wavelengths, 1, num_pts_y, num_pts_x), dtype=dtype)
    holdField_phase = tf.zeros((num_wavelengths, 1, num_pts_y, num_pts_x), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdField_ampl, holdField_phase, num_ms])

    return (
        tf.stack(loopData[1][:, 1:, :, :]),
        tf.stack(loopData[2][:, 1:, :, :]),
    )


def batch_field_propagation_MatrixASM(field_amplitude, field_phase, sim_wavelengths_m, modified_parameters):
    """Helper function to batch the field propagation via Matrix ASM method over the profile batch dimension. Wavelength
    calculations are done efficiently without loops. If one does not want to batch profiles, just call the field propagation
    method alone.

    Args:
        `field_amplitude` (tf.float): Initial plane field amplitude, of shape
            (len(sim_wavelengths) or 1, batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (len(sim_wavelengths) or 1, batch_size, 1, ms_samplesM["r"])
        `field_phase` (tf.float): Initial plane field phase, the same shape as field_amplitude
        `sim_wavelengths_m` (tf.float): List of simulation wavelengths to propagate the field with
        `modified_parameters` (tf.float): Modified propagation parameters, like generated in Propagate_Planes_Layer_MatrixBroadband

    Returns:
        `tf.float`: Output plane field amplitude, of shape
            ( len(sim_wavelengths), batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"] )
        `tf.float`: Output plane field phase, of shape
            ( len(sim_wavelengths), batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # unpack parameters
    num_ms = field_amplitude.shape[1]

    # define loop condition
    def lambda_loopCond(idx_, holdField_ampl_, holdField_phase_):
        return tf.less(idx_, num_ms)

    # The loop body
    def lambda_loopBody(idx_, holdField_ampl_, holdField_phase_):
        ampl, phase = field_propagation_MatrixASM(
            field_amplitude[:, idx_ : idx_ + 1, :, :],
            field_phase[:, idx_ : idx_ + 1, :, :],
            sim_wavelengths_m,
            modified_parameters,
        )

        holdField_ampl_ = tf.concat([holdField_ampl_, ampl], axis=1)
        holdField_phase_ = tf.concat([holdField_phase_, phase], axis=1)

        idx_ += 1

        return [idx_, holdField_ampl_, holdField_phase_]

    # Run batched loop
    num_wavelengths = len(sim_wavelengths_m)
    sensor_pixel_number = modified_parameters["sensor_pixel_number"]
    radial_symmetry = modified_parameters["radial_symmetry"]
    num_pts_x = tf.cond(radial_symmetry, lambda: sensor_pixel_number["r"], lambda: sensor_pixel_number["x"])
    num_pts_y = tf.cond(radial_symmetry, lambda: 1, lambda: sensor_pixel_number["y"])
    dtype = modified_parameters["dtype"]
    idx = tf.constant(0, dtype=tf.int32)

    holdField_ampl = tf.zeros((num_wavelengths, 1, num_pts_y, num_pts_x), dtype=dtype)
    holdField_phase = tf.zeros((num_wavelengths, 1, num_pts_y, num_pts_x), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdField_ampl, holdField_phase])

    return (
        tf.stack(loopData[1][:, 1:, :, :]),
        tf.stack(loopData[2][:, 1:, :, :]),
    )


### PSF Layer Routines
def loopBatch_psf_measured(point_source_locs, ms_trans, ms_phase, parameters, normby):
    """Loops the psf measured routine over batch. This was written for the PSF_Layer_Mono, i.e. monochromatic.

    We don't need to do any wavelength looping and although loopWavelength_psf_measured can be called, it is simple
    enough just to write a different loop.

    Args:
        `point_source_locs` (tf.float): Tensor of point-source coordinates, of shape (N,3)
        `ms_trans` (tf.float): Transmittance profile of the optical metasurface, of shape
            (profile_batch, ms_samplesM['y'], ms_samplesM['x']) or (profile_batch, 1, ms_samplesM['r']).
        `ms_phase` (tf.float):  Phase profile of the optical metasurface, same shape as ms_trans
        parameters (prop_params): Propagation parameters setting object (wavelength_m defined)
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
    """

    # unpack parameters
    num_ms = ms_trans.shape[0]

    # define loop condition
    def lambda_loopCond(idx_, holdPSF_int_, holdPSF_phase_):
        return tf.less(idx_, num_ms)

    # The loop body
    def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
        psfs_int, psfs_phase = psf_measured(point_source_locs, ms_trans[idx_ : idx_ + 1], ms_phase[idx_ : idx_ + 1], parameters, normby)

        holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=0)
        holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=0)

        idx_ += 1

        return [idx_, holdPSF_int_, holdPSF_phase_]

    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]
    num_ps = point_source_locs.shape[0]
    idx = tf.constant(0, dtype=tf.int32)

    holdPSF_int = tf.zeros((1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    holdPSF_phase = tf.zeros((1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdPSF_int, holdPSF_phase])

    return (
        tf.stack(loopData[1][1:, :, :, :]),
        tf.stack(loopData[2][1:, :, :, :]),
    )


def loopWavelength_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters_list):
    """Loops the PSF measured routine over wavelength. If one wants to compute over wavelengths without batching, the X
    caller should be used instead which implements a matrix broadband calculation using the ASM engine.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `ms_trans` (tf.float): Metasurface transmittance profiles, of shape
            (len(wavelength_set_m), num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), num_profiles, 1, ms_samplesM["r"]). Alternatively, the transmittance of the optical
            element may be assumed wavelength-independent and given rank 3 tensor of shape
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
    input_rank = tf.rank(ms_trans)

    # define loop condition
    def lambda_loopCond(idx_, holdPSF_int_, holdPSF_phase_):
        return tf.less(idx_, num_wavelengths)

    # define loop body dependent on overloading input of rank 3 vs rank 4
    if input_rank == tf.TensorShape(3):
        # the metasurface modulation profiles are the same across wavelength
        num_ms = ms_trans.shape[0]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
            psfs_int, psfs_phase = psf_measured(
                point_source_locs,
                ms_trans,
                ms_phase,
                parameters_list[idx_],
                normby,
            )
            holdPSF_int_ = tf.concat([holdPSF_int_, tf.expand_dims(psfs_int, 0)], axis=0)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, tf.expand_dims(psfs_phase, 0)], axis=0)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_]

    elif input_rank == tf.TensorShape(4):
        # The metasurface modulations are defined for each wavelength channel
        num_ms = ms_trans.shape[1]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
            psfs_int, psfs_phase = psf_measured(
                point_source_locs,
                ms_trans[idx_],
                ms_phase[idx_],
                parameters_list[idx_],
                normby,
            )
            holdPSF_int_ = tf.concat([holdPSF_int_, tf.expand_dims(psfs_int, 0)], axis=0)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, tf.expand_dims(psfs_phase, 0)], axis=0)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_]

    else:
        raise ValueError("broadband_batched_psf_measured: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    # Run batched loop with manual conditional overloading on function
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]  # wavelength invariant quantity so grab from any
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


def batch_loopWavelength_psf_measured(ms_trans, ms_phase, normby, point_source_locs, parameters_list):
    """Loops the PSF measured routine over wavelength and also adds a batch loop over the metasurface profiles dimension.
    To not explicitly batch over metasurfaces, use loopWavelength_psf_measured instead. See there for more details.

    When the metasurface profile is a rank 4 tensor, the user asserts that the modulation functions differ across
    wavelength (as is true for real metasurfaces). If a rank 3 tensor, the set of modulation profiles are assumed to
    be the same across wavelength and are thus need not be trivially redefined in the input. The PSF in either case
    is still wavelength-dependent (diffraction propagation is inherrently wavelength-dependent).
    The output structure is the same but this function is overloaded on input.

    Args:
        `ms_trans` (tf.float): Metasurface transmittance profiles, of shape
            (len(wavelength_set_m), num_profiles, ms_samplesM["y"], ms_samplesM["x"]) or
            (len(wavelength_set_m), num_profiles, 1, ms_samplesM["r"]). Alternatively, the transmittance of the optical
            element may be assumed wavelength-independent and given rank 3 tensor of shape
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
    input_rank = tf.rank(ms_trans)
    num_wavelengths = len(parameters_list)

    # define loop condition
    def lambda_loopCond(idx_, holdPSF_int_, holdPSF_phase_, num_ms_):
        return tf.less(idx_, num_ms_)

    # define loop body dependent on overloading input of rank 3 vs rank 4
    if input_rank == tf.TensorShape(3):
        # the metasurface modulation profiles are the same across wavelength
        num_ms = ms_trans.shape[0]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_, num_ms_):
            psfs_int, psfs_phase = loopWavelength_psf_measured(
                ms_trans[idx_ : idx_ + 1], ms_phase[idx_ : idx_ + 1], normby, point_source_locs, parameters_list
            )
            holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=1)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=1)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_, num_ms_]

    elif input_rank == tf.TensorShape(4):
        # The metasurface modulations are defined for each wavelength channel
        num_ms = ms_trans.shape[1]

        def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_, num_ms_):
            psfs_int, psfs_phase = loopWavelength_psf_measured(
                ms_trans[:, idx_ : idx_ + 1, :, :],
                ms_phase[:, idx_ : idx_ + 1, :, :],
                normby,
                point_source_locs,
                parameters_list,
            )
            holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=1)
            holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=1)

            idx_ += 1

            return [idx_, holdPSF_int_, holdPSF_phase_, num_ms_]

    else:
        raise ValueError("broadband_batched_psf_measured: rank of ms_trans and/or ms_phase is incorrect. must be rank 3 or rank 4 tensor.")

    # Run batched loop with manual conditional overloading on function
    sensor_pixel_number = parameters_list[0]["sensor_pixel_number"]  # wavelength invariant quantity so grab from any
    dtype = parameters_list[0]["dtype"]
    num_ps = point_source_locs.shape[0]
    idx = tf.constant(0, dtype=tf.int32)

    holdPSF_int = tf.zeros((num_wavelengths, 1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    holdPSF_phase = tf.zeros((num_wavelengths, 1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdPSF_int, holdPSF_phase, num_ms])

    return (
        tf.stack(loopData[1][:, 1:, :, :, :]),
        tf.stack(loopData[2][:, 1:, :, :, :]),
    )


def batch_psf_measured_MatrixASM(
    sim_wavelengths_m,
    point_source_locs,
    ms_trans,
    ms_phase,
    modified_parameters,
    normby,
):
    """Loops the psf_measured_Matrix_ASM routine by for-loop batching over the ms_batch dimension.

    Args:
        `sim_wavelengths_m` (tf.float): Tensor of simulation wavelengths to compute the psf for
        `point_source_locs` (tf.float): Tensor of shape (Nps, 3) corresponding to point-source locations to compute psf for
        `ms_trans` (tf.float): Transmittance profile(s) of the optical metasurface(s), of shape
            (len(wavelength_set_m) or 1, profile_batch, ms_samplesM['y'], ms_samplesM['x']),
            or (len(wavelength_set_m) or 1, profile_batch, 1, ms_samplesM['r']).
        `ms_phase` (tf.float): Phase profiles of the metasurface, same shape as ms_trans
        `modified_parameters` (prop_params): Modified prop_params as generated in PSF_Layer_MatrixBroadband
        `normby` (tf.float): Scalar tensor corresponding to a normalization factor to dive the computed psf by,
            (ideally to make the total psf energy unity).

    Returns:
        `tf.float`: PSF intensity of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float`: PSF phase of shape (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    # unpack parameters
    num_ms = ms_trans.shape[1]

    # define loop condition
    def lambda_loopCond(idx_, holdPSF_int_, holdPSF_phase_):
        return tf.less(idx_, num_ms)

    # The loop body
    def lambda_loopBody(idx_, holdPSF_int_, holdPSF_phase_):
        psfs_int, psfs_phase = psf_measured_MatrixASM(
            sim_wavelengths_m,
            point_source_locs,
            ms_trans[:, idx_ : idx_ + 1, :, :],
            ms_phase[:, idx_ : idx_ + 1, :, :],
            modified_parameters,
            normby,
        )

        holdPSF_int_ = tf.concat([holdPSF_int_, psfs_int], axis=1)
        holdPSF_phase_ = tf.concat([holdPSF_phase_, psfs_phase], axis=1)

        idx_ += 1

        return [idx_, holdPSF_int_, holdPSF_phase_]

    sensor_pixel_number = modified_parameters["sensor_pixel_number"]
    dtype = modified_parameters["dtype"]
    num_ps = point_source_locs.shape[0]
    idx = tf.constant(0, dtype=tf.int32)
    num_wl = len(sim_wavelengths_m)

    holdPSF_int = tf.zeros((num_wl, 1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    holdPSF_phase = tf.zeros((num_wl, 1, num_ps, sensor_pixel_number["y"], sensor_pixel_number["x"]), dtype=dtype)
    loopData = tf.while_loop(lambda_loopCond, lambda_loopBody, loop_vars=[idx, holdPSF_int, holdPSF_phase])

    return (
        tf.stack(loopData[1][:, 1:, :, :, :]),
        tf.stack(loopData[2][:, 1:, :, :, :]),
    )
