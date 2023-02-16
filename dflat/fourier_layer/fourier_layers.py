import tensorflow as tf
import numpy as np

from dflat.data_structure import prop_params
from .core.ops_field_aperture import gen_aperture_disk
from .core.caller_BatchedFourierOpt import *


def check_broadband_wavelength_parameters(parameters):
    if not ("wavelength_set_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_set_m")

    return


def check_single_wavelength_parameters(parameters):
    if not ("wavelength_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_m")

    if not (type(parameters["wavelength_m"]) == float):
        raise ValueError("Wavelength should be a single float")
    return


## PSF Layers
class PSF_Layer(tf.keras.layers.Layer):
    """Fourier optics-based, point-spread function computing instance (single prop_param setting configuration).
    Computes the psf of the optical system for multiple wavelengths given a set of metasurface modulation profile(s) and
    a set of point-source distances. The metasurface profiles may be defied once to be the same for all wavelength channels
    or may be defined explicitly for each wavelength.

    Once initialized for a given geometry and grid, it may be called repeatedly. 'wavelength_set_m' must be defined.
    This implementation uses a for loop over the wavelengths; It includes a helper to optionally batch/loop metasurface
    profiles as well.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `parameters_list` (list of prop_params objects): A list of prop_param configuration objects
            generated at run-time for each wavelength in the set.
        `aperture_trans` (tf.float64): Pre-metasurface field aperture used in calculation, of shape
            (1, ms_samplesM["y"], ms_samplesM["x"]).
    """

    def __init__(self, parameters):
        """Fourier PSF Layer Initialization.

        Args:
            `parameters` (prop_param): Settings object defining field propagation details. The set of wavelengths for
                the calculation is defined by key 'wavelength_set_m'.

        Raises:
            KeyError: 'wavelength_set_m' must be defined in the parameters object.
        """

        super(PSF_Layer, self).__init__()
        self.parameters = parameters
        check_broadband_wavelength_parameters(parameters)

        # Generate the Fourier grids for each wavelength which is defined in each of the new parameter objects
        self.parameters_list = self.__generate_simParam_set()

        # Define the energy normalization factor
        aperture_trans, sqrt_energy_illum = gen_aperture_disk(parameters)
        self.__sqrt_energy_illum = tf.convert_to_tensor(sqrt_energy_illum, dtype=parameters["dtype"])
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

    def __call__(self, inputs, point_source_locs, batch_loop=False):
        """The broadband psf_compute call function. Computes the PSF, given a set of point_source_locs and a set of phase
        and transmittance profiles for each wavelength in the set. This call enables overloading, such that the
        metasurface profiles may be uniquely defined for each wavelength or assumed to be the same across wavelength
        channels.

        The profile_batch dimension can be used to describe a polarization-sensitive metasurface, via a stacked pair of phase
        and transmittance profiles be passed in at once, corresponding to the optical response on two orthogonal, polarization
        basis states for each wavelength channel in the set. The profile batch dimension may be more generally used to represent
        the phase and transmittance for a general set of many polarization states or to represent a set of many different
        metasurfaces at each wavelength.

        Args:
            `inputs` (list): Input arguments which consist of ms_trans in first arg and ms_phase in second:
                transmittance` (float): Transmittance profile(s) of the optical metasurface(s), of shape
                (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']),
                or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                same across wavelength, one may pass in transmittance of shape without wavelength dimension via,
                (profile_batch, ms_samplesM['y'], ms_samplesM['x']) or (profile_batch, 1, ms_samplesM['r']).

                phase (float): Phase profile(s) of the optical metasurface(s), of shape
                (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']).
                or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                same across wavelength, one may pass in phase of shape, (profile_batch, ms_samplesM['y'], ms_samplesM['x'])
                or (profile_batch, 1, ms_samplesM['r']).

            `point_source_locs` (float): Tensor of point-source coordinates, of shape (N,3).

            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once

        Returns:
            `list`: List containing the detector measured PSF intensity in the first argument and the phase in the
                second argument, of shape
                (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        """
        use_dtype = self.parameters["dtype"]
        ms_trans = inputs[0]
        ms_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(ms_trans):
            ms_trans = tf.convert_to_tensor(ms_trans, dtype=use_dtype)
        else:
            if not ms_trans.dtype == use_dtype:
                ms_trans = tf.cast(ms_trans, use_dtype)

        if not tf.is_tensor(ms_phase):
            ms_phase = tf.convert_to_tensor(ms_phase, dtype=use_dtype)
        else:
            if not ms_phase.dtype == use_dtype:
                ms_phase = tf.cast(ms_phase, use_dtype)

        if not tf.is_tensor(point_source_locs):
            point_source_locs = tf.convert_to_tensor(point_source_locs, dtype=self.parameters["dtype"])

        # Apply the metasurface aperture
        ms_rank = tf.rank(ms_trans)
        if ms_rank == tf.TensorShape(3):
            ms_trans = ms_trans * self.aperture_trans
        elif ms_rank == tf.TensorShape(4):
            ms_trans = ms_trans * tf.expand_dims(self.aperture_trans, 0)

        if batch_loop:
            out = batch_loopWavelength_psf_measured(ms_trans, ms_phase, self.__sqrt_energy_illum, point_source_locs, self.parameters_list)
        else:
            out = loopWavelength_psf_measured(ms_trans, ms_phase, self.__sqrt_energy_illum, point_source_locs, self.parameters_list)

        return out

    def __generate_simParam_set(self):
        wavelength_set_m = self.parameters["wavelength_set_m"]
        parameters_list = []

        for wavelength in wavelength_set_m:
            setting_dict = self.parameters.get_dict()
            del setting_dict["wavelength_set_m"]
            setting_dict["wavelength_m"] = wavelength
            parameters_list.append(prop_params(setting_dict))

        return parameters_list


class PSF_Layer_MatrixBroadband(tf.keras.layers.Layer):
    """Fourier optics-based, point-spread function computing instance (single prop_param setting configuration).
    Computes the psf of the optical system for multiple wavelengths given a set of metasurface modulation profile(s) and
    a set of point-source distances. The metasurface profiles may be defied once to be the same for all wavelength channels
    or may be defined explicitly for each wavelength.

    Once initialized for a given geometry and grid, it may be called repeatedly. This is a special routine which leverages
    the structure of the ASM propagation calculation to enable broadband calculations without for loops. The broadband calculation
    can instead be done with higher rank tensor operations enabling substantial run-time speedup when the number of wavelengths
    to simulate with is large. No wavelength batching is allowed here since the point is to accelerate computation time.
    This function includes a helper to optionally batch/loop the metasurface profiles.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `modified_parameters` (prop_params): A new prop_param configuration object generated at run-time. It chooses the
            calculation sampling and grid requirements based on the strictes condition. This prop_params will be used for all
            wavelength channels in the calculation.
        `aperture_trans` (tf.float64): Pre-metasurface field aperture used in calculation, of shape
            (1, ms_samplesM["y"], ms_samplesM["x"]).
    """

    def __init__(self, parameters):
        """Fourier PSF Layer with Accelerated broadband calculations (ASM fourier only)

        Args:
            parameters (prop_param): Settings object defining field propagation details. The set of wavelengths for
                the calculation is defined by key 'wavelength_set_m'. This can be overrrided later but this wavelength range
                is used to define the calculation grid. Note; ASM_fourier must be chosen as the diffraction engine.
        """
        super(PSF_Layer_MatrixBroadband, self).__init__()

        self.parameters = parameters
        check_broadband_wavelength_parameters(parameters)
        self.wavelength_set_m = parameters["wavelength_set_m"]

        # Verify that the user selected ASM_fourier engine as that is the only option for this accelerated routine
        if parameters["diffractionEngine"] != "ASM_fourier":
            raise ValueError("This PSF Layer is only valid for ASM_fourier diffraction engine. Fresnel cannot be used here!")

        # generate the modified parameters object suitable for the ASM broadband matrix version
        self.modified_parameters = self.__generate_new_parameters()

        aperture_trans, sqrt_energy_illum = gen_aperture_disk(parameters)
        self.__sqrt_energy_illum = tf.convert_to_tensor(sqrt_energy_illum, dtype=parameters["dtype"])
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

        return

    def __call__(self, inputs, point_source_locs, sim_wavelengths_m=None, batch_loop=False):
        """The broadband psf_compute call function. Computes the PSF, given a set of point_source_locs and a set of phase
        and transmittance profiles for each wavelength in the set. This call enables overloading, such that the
        metasurface profiles may be uniquely defined for each wavelength or assumed to be the same across wavelength
        channels.

        The profile_batch dimension can be used to describe a polarization-sensitive metasurface, via a stacked pair of phase
        and transmittance profiles be passed in at once, corresponding to the optical response on two orthogonal, polarization
        basis states for each wavelength channel in the set. The profile batch dimension may be more generally used to represent
        the phase and transmittance for a general set of many polarization states or to represent a set of many different
        metasurfaces at each wavelength.

        Args:
            `inputs` (list): len(2) list containing metasurface transmission in first argument and phase in second argument:
                o `ms_trans` (float): Transmittance profile(s) of the optical metasurface(s), of shape
                    (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']),
                    or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                    same across wavelength, one may pass in transmittance of shape,
                    (profile_batch, ms_samplesM['y'], ms_samplesM['x']) or  (profile_batch, 1, ms_samplesM['r']).

                o `ms_phase` (float): Phase profile(s) of the optical metasurface(s), of shape
                    (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']).
                    or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                    same across wavelength, one may pass in phase of shape, (profile_batch, ms_samplesM['y'], ms_samplesM['x'])
                    or (profile_batch, 1, ms_samplesM['r']).

            `point_source_locs` (float): Tensor of point-source coordinates, of shape (N,3).

            `sim_wavelengths_m` (array, defaults to None): List of wavelengths to simulate the psf for. If None, the wavelengths
                passed during initialization (in parameters) will be used.

            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once

            Returns:
            `list`: List containing the detector measured PSF intensity in the first argument and the phase in the
                second argument, of shape
                (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        """
        use_dtype = self.parameters["dtype"]
        ms_trans = inputs[0]
        ms_phase = inputs[1]

        # New simulation values may be passed in that override the ones used in parameters
        # The reason for this might be random variations during training iterations to encourage learned generality
        if sim_wavelengths_m is None:
            sim_wavelengths_m = self.wavelength_set_m

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(ms_trans):
            ms_trans = tf.convert_to_tensor(ms_trans, dtype=self.parameters["dtype"])
        else:
            if not ms_trans.dtype == use_dtype:
                ms_trans = tf.cast(ms_trans, use_dtype)

        if not tf.is_tensor(ms_phase):
            ms_phase = tf.convert_to_tensor(ms_phase, dtype=self.parameters["dtype"])
        else:
            if not ms_phase.dtype == use_dtype:
                ms_phase = tf.cast(ms_phase, use_dtype)

        if not tf.is_tensor(point_source_locs):
            point_source_locs = tf.convert_to_tensor(point_source_locs, dtype=self.parameters["dtype"])

        if not tf.is_tensor(sim_wavelengths_m):
            sim_wavelengths_m = tf.convert_to_tensor(sim_wavelengths_m, dtype=self.parameters["dtype"])

        # Apply the metasurface aperture
        # Also if metasurface tensors are rank 3, convert to rank 4
        ms_rank = tf.rank(ms_trans)
        if ms_rank == tf.TensorShape(3):
            ms_trans = tf.expand_dims(ms_trans * self.aperture_trans, 0)
            ms_phase = tf.expand_dims(ms_phase, 0)
        elif ms_rank == tf.TensorShape(4):
            ms_trans = ms_trans * tf.expand_dims(self.aperture_trans, 0)

        if batch_loop:
            out = batch_psf_measured_MatrixASM(
                sim_wavelengths_m,
                point_source_locs,
                ms_trans,
                ms_phase,
                self.modified_parameters,
                self.__sqrt_energy_illum,
            )
        else:
            out = psf_measured_MatrixASM(
                sim_wavelengths_m,
                point_source_locs,
                ms_trans,
                ms_phase,
                self.modified_parameters,
                self.__sqrt_energy_illum,
            )

        return out

    def __generate_new_parameters(self):
        # For this ASM-based subroutine, we want a single regularized parameters that enforces the strictest sampling condition
        # The largest fourier bandwidth occurs for smallest wavelengths
        # min_wavelength = np.min(self.wavelength_set_m)
        min_wavelength = np.min(self.wavelength_set_m)
        setting_dict = self.parameters.get_dict()
        del setting_dict["wavelength_set_m"]
        setting_dict["wavelength_m"] = min_wavelength

        # Calling the prop_params with the new wavelength will automatically run sub-routines to compute
        # proper resizing factors and upsampling requirements.
        modified_parameters = prop_params(setting_dict)

        # The downstream routines will be doing broadband calculations and will utilize a seperate wavelength_set_m input
        # It will not use wavelength_m in the prop_params for this code hidden to users. For safety, let us flip
        # the wavelength value now to None to avoid possible mistakes or misuse
        modified_parameters._set_wavelength_m_None()

        return modified_parameters


class PSF_Layer_Mono(tf.keras.layers.Layer):
    """Fourier optics-based, point-spread function computing instance (single prop_param setting configuration and
    single wavelength). Computes the psf(s) of the optical system for different point-source(s), given metasurface
    modulation profile(s)--transmittance and phase--on a single wavelength.

    Once initialized for a given geometry and grid, it may be called repeatedly. For psfs as a function of many
    wavelengths, use psf_broadband_layer instead. This function is provided as it is simpler than the broadband case
    and can be used for basic demos or tests, or slightly faster monochromatic rendering.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `aperture_trans` (tf.float): Pre-metasurface field aperture used in calculation, of shape
            (1, ms_samplesM["y"], ms_samplesM["x"]).
    """

    def __init__(self, parameters):
        """Fourier PSF Layer Initialization (monochromatic)

        Args:
            `parameters` (prop_param): Settings object defining field propagation details. Wavelength for calculation
                is set by parameters["wavelength_m"].

        Raises:
            KeyError: parameters object must have 'wavelength_m' defined.
            ValueError: The 'wavelength_m' value must be a single float.
        """

        super(PSF_Layer_Mono, self).__init__()
        self.parameters = parameters
        check_single_wavelength_parameters(parameters)

        aperture_trans, sqrt_energy_illum = gen_aperture_disk(parameters)
        self.__sqrt_energy_illum = tf.convert_to_tensor(sqrt_energy_illum, dtype=parameters["dtype"])
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

    def __call__(self, inputs, point_source_locs, batch_loop=False):
        """The psf_layer call function. Computes the PSF, given a set of point_source_locs and the set of phase and
        transmittance profiles.

        For a polarization-sensitive metasurface, a stacked pair of phase and transmittance profiles may be passed in
        at once, corresponding to the optical response on orthogonal, polarization basis states. The metasurface
        batch dimension may be more generally used to represent the phase and transmittance modulation for a set of many
        polarization states or to represent a set of many different metasurfaces.

        Args:
            `inputs` (list): len(2) list containing metasurface transmission in first argument and phase in second argument:
                o `ms_trans` (float): Transmittance profile of the optical metasurface, of shape
                    (profile_batch, ms_samplesM['y'], ms_samplesM['x']) or (profile_batch, 1, ms_samplesM['r']).

                o `ms_phase` (float): Phase profile of the optical metasurface, same shape as ms_trans

            `point_source_locs` (float): Tensor of point-source coordinates, of shape (N,3)

            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once

        Returns:
            `list`: List containing the detector measured PSF intensity in the first argument and the phase in the
                second argument, of shape (profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"]).
        """
        use_dtype = self.parameters["dtype"]
        ms_trans = inputs[0]
        ms_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(ms_trans):
            ms_trans = tf.convert_to_tensor(ms_trans, dtype=use_dtype)
        else:
            if not ms_trans.dtype == use_dtype:
                ms_trans = tf.cast(ms_trans, use_dtype)

        if not tf.is_tensor(ms_phase):
            ms_phase = tf.convert_to_tensor(ms_phase, dtype=self.parameters["dtype"])
        else:
            if not ms_phase.dtype == use_dtype:
                ms_phase = tf.cast(ms_phase, use_dtype)

        if not tf.is_tensor(point_source_locs):
            point_source_locs = tf.convert_to_tensor(point_source_locs, dtype=self.parameters["dtype"])

        # Check if convenient input rank is used (Rank 4 with size 1 extra dimension)
        ms_rank = tf.rank(ms_trans)
        if ms_rank == tf.TensorShape(4):
            ms_trans = tf.squeeze(ms_trans, 0)
            ms_phase = tf.squeeze(ms_phase, 0)

        # Apply the metasurface aperture
        ms_trans = ms_trans * self.aperture_trans

        if batch_loop:
            out = loopBatch_psf_measured(point_source_locs, ms_trans, ms_phase, self.parameters, self.__sqrt_energy_illum)
        else:
            out = psf_measured(
                point_source_locs,
                ms_trans,
                ms_phase,
                self.parameters,
                self.__sqrt_energy_illum,
            )

        return out


## Field Propagation Layers
class Propagate_Planes_Layer(tf.keras.layers.Layer):
    """Fourier optics-based field propagator instance (reuses prop_param configurations to define input and output
    grids and distances). Computes the output field(s) a fixed distance away from an initial plane, given the complex
    fields on a set of wavelength channels.

    Once initialized for a given geometry and grid, it may be called repeatedly. 'wavelength_set_m' must be defined and
    is expected to be the same for all fields.
    - Plane seperation distance is defined by parameters["sensor_distance_m"], regardless of if the output plane is the
        sensor or an intermediate plane.
    - The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    - The output grid is defined by parameters["sensor_dx_m"] and parameters["sensor_pixel_number"].
    Thus, the input grid and output grid can be fully specified by the user.


    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `parameters_list` (list of prop_params objects): A list of prop_param configuration objects initialized for
            each wavelength in the set.
    """

    def __init__(self, parameters):
        """propagate_plane_layer initialization.

        Args:
            `parameters` (prop_param): Settings object defining field propagation details. Wavelength set for
                calculation is defined by parameters["wavelength_set_m"].

        Raises:
            KeyError: parameters object must have 'wavelength_set_m' defined.
        """
        super(Propagate_Planes_Layer, self).__init__()
        self.parameters = parameters
        check_broadband_wavelength_parameters(parameters)

        # Generate the fourier grids for each wavelength
        self.parameters_list = self.__generate_simParam_set()

        # Allow for application of aperture like with psf propagation case
        aperture_trans, _ = gen_aperture_disk(parameters)
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

    def __call__(self, inputs, batch_loop=False):
        """propagate_planes_broadband_layer call function. Computes the field amplitude and phase at a parallel plane a
        distance away from the initial plane, for multiple wavelength channels.

        A batch of field profiles (optionally, a batch specified for each wavelength channel) may be passed in and
        computed at once. The field profile batches may often represent the field for different polarization states of
        light or more generally, may represent the fields from different devices, at each wavelength.

        Args:
            `inputs` (list): Length two list containing field amplitude in first arg and field phase in second:
                o `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
                    (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
                    (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
                    is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
                    or (profile_batch, 1, ms_samplesM["r"]).
                o `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
                    (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
                    (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
                    is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
                    or (batch_size, 1, ms_samplesM["r"]).

            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once

        Returns:
            `list`: List of field amplitude(s) in the first argument and phase(s) in the second arg at the output plane.
            The shape of each is given via (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        """
        use_dtype = self.parameters["dtype"]
        field_amplitude = inputs[0]
        field_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(field_amplitude):
            field_amplitude = tf.convert_to_tensor(field_amplitude, dtype=use_dtype)
        else:
            if not field_amplitude.dtype == use_dtype:
                field_amplitude = tf.cast(field_amplitude, use_dtype)

        if not tf.is_tensor(field_phase):
            field_phase = tf.convert_to_tensor(field_phase, dtype=self.parameters["dtype"])
        else:
            if not field_phase.dtype == use_dtype:
                field_phase = tf.cast(field_phase, use_dtype)

        # Apply the metasurface aperture
        ms_rank = tf.rank(field_amplitude)
        if ms_rank == tf.TensorShape(3):
            field_amplitude = field_amplitude * self.aperture_trans
        elif ms_rank == tf.TensorShape(4):
            field_amplitude = field_amplitude * tf.expand_dims(self.aperture_trans, 0)

        if batch_loop:
            out = batch_loopWavelength_field_propagation(field_amplitude, field_phase, self.parameters_list)
        else:
            out = loopWavelength_field_propagation(field_amplitude, field_phase, self.parameters_list)

        return out

    def __generate_simParam_set(self):
        wavelength_set_m = self.parameters["wavelength_set_m"]
        parameters_list = []

        for wavelength in wavelength_set_m:
            setting_dict = self.parameters.get_dict()
            del setting_dict["wavelength_set_m"]
            setting_dict["wavelength_m"] = wavelength
            parameters_list.append(prop_params(setting_dict))

        return parameters_list


class Propagate_Planes_Layer_MatrixBroadband(tf.keras.layers.Layer):
    """Fourier optics-based field propagator instance (reuses prop_param configurations to define input and output
    grids and distances). Computes the output field(s) a fixed distance away from an initial plane, given the complex
    fields on a set of wavelength channels.

    Once initialized for a given geometry and grid, it may be called repeatedly. 'wavelength_set_m' must be defined and
    is expected to be the same for all fields.
    - Plane seperation distance is defined by parameters["sensor_distance_m"], regardless of if the output plane is the
        sensor or an intermediate plane.
    - The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    - The output grid is defined by parameters["sensor_dx_m"] and parameters["sensor_pixel_number"].
    Thus, the input grid and output grid can be fully specified by the user.

    This is a special routine which leverages
    the structure of the ASM propagation calculation to enable broadband calculations without for loops. The broadband calculation
    can instead be done with higher rank tensor operations enabling substantial run-time speedup when the number of wavelengths
    to simulate with is large. No wavelength batching is allowed here since the point is to accelerate computation time.
    This function includes a helper to optionally batch/loop the metasurface profiles.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `modified_parameters` (prop_params): A new prop_param configuration object generated at run-time. It chooses the
            calculation sampling and grid requirements based on the strictes condition. This prop_params will be used for all
            wavelength channels in the calculation.
    """

    def __init__(self, parameters):
        """propagate_plane_layer initialization.

        Args:
            `parameters` (prop_param): Settings object defining field propagation details. Wavelength set for
                calculation is defined by parameters["wavelength_set_m"].
        """
        super(Propagate_Planes_Layer_MatrixBroadband, self).__init__()

        self.parameters = parameters
        check_broadband_wavelength_parameters(parameters)
        self.wavelength_set_m = parameters["wavelength_set_m"]

        # Verify that the user selected ASM_fourier engine as that is the only option for this accelerated routine
        if parameters["diffractionEngine"] != "ASM_fourier":
            raise ValueError("This Propagation Layer is only valid for ASM_fourier diffraction engine. Fresnel cannot be used here!")

        # generate the modified parameters object suitable for the ASM broadband matrix version
        self.modified_parameters = self.__generate_new_parameters()

        # Allow for application of aperture like with psf propagation case
        aperture_trans, _ = gen_aperture_disk(parameters)
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

        return

    def __call__(self, inputs, sim_wavelengths_m=None, batch_loop=False):
        """Propagate_Planes_Layer_MatrixBroadband call function. Computes the field amplitude and phase at a parallel plane a
        distance away from the initial plane, for multiple wavelength channels.

        A batch of field profiles (optionally, a batch specified for each wavelength channel) may be passed in and
        computed at once. The field profile batches may often represent the field for different polarization states of
        light or more generally, may represent the fields from different devices, at each wavelength.

        Args:
            `inputs` (list): Length two list containing field amplitude in first arg and field phase in second:
                o `field_amplitude` (tf.float): Amplitude(s) at the initial plane, in shape of
                    (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
                    (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field amplitude
                    is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
                    or (profile_batch, 1, ms_samplesM["r"]).
                o `field_phase` (tf.float): Phase(s) at the initial plane, in shape of
                    (len(wavelength_set_m), profile_batch, ms_samplesM["y"], ms_samplesM["x"]) or
                    (len(wavelength_set_m), profile_batch, 1, ms_samplesM["r"]). Alternatively, if the field phase
                    is the same across wavelength channels, shape may be (profile_batch, ms_samplesM["y"], ms_samplesM["x"])
                    or (batch_size, 1, ms_samplesM["r"]).

            `sim_wavelengths_m` (tf.float, defaults to None): List of wavelengths to simulate the psf for. If None, the wavelengths
                passed during initialization (in parameters) will be used.

            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once
        """
        use_dtype = self.parameters["dtype"]
        field_amplitude = inputs[0]
        field_phase = inputs[1]

        # New simulation values may be passed in that override the ones used in parameters
        # The reason for this might be random variations during training iterations to encourage learned generality
        if sim_wavelengths_m is None:
            sim_wavelengths_m = self.wavelength_set_m

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(field_amplitude):
            field_amplitude = tf.convert_to_tensor(field_amplitude, dtype=use_dtype)
        else:
            if not field_amplitude.dtype == use_dtype:
                field_amplitude = tf.cast(field_amplitude, use_dtype)

        if not tf.is_tensor(field_phase):
            field_phase = tf.convert_to_tensor(field_phase, dtype=self.parameters["dtype"])
        else:
            if not field_phase.dtype == use_dtype:
                field_phase = tf.cast(field_phase, use_dtype)

        if not tf.is_tensor(sim_wavelengths_m):
            sim_wavelengths_m = tf.convert_to_tensor(sim_wavelengths_m, dtype=self.parameters["dtype"])

        # If metasurface tensors are rank 3, convert to rank 4
        ms_rank = tf.rank(field_amplitude)
        if ms_rank == tf.TensorShape(3):
            field_amplitude = tf.expand_dims(field_amplitude, 0)
            field_phase = tf.expand_dims(field_phase, 0)

        # Apply the field aperture if requested
        field_amplitude = field_amplitude * tf.expand_dims(self.aperture_trans, 0)

        if batch_loop:
            out = batch_field_propagation_MatrixASM(field_amplitude, field_phase, sim_wavelengths_m, self.modified_parameters)
        else:
            out = field_propagation_MatrixASM(field_amplitude, field_phase, sim_wavelengths_m, self.modified_parameters)

        return out

    def __generate_new_parameters(self):
        # For this ASM-based subroutine, we want a single regularized parameters that enforces the strictest sampling condition
        # The largest fourier bandwidth occurs for smallest wavelengths
        # min_wavelength = np.min(self.wavelength_set_m)
        min_wavelength = np.min(self.wavelength_set_m)
        setting_dict = self.parameters.get_dict()
        del setting_dict["wavelength_set_m"]
        setting_dict["wavelength_m"] = min_wavelength

        # Calling the prop_params with the new wavelength will automatically run sub-routines to compute
        # proper resizing factors and upsampling requirements.
        modified_parameters = prop_params(setting_dict)

        # The downstream routines will be doing broadband calculations and will utilize a seperate wavelength_set_m input
        # It will not use wavelength_m in the prop_params for this code hidden to users. For safety, let us flip
        # the wavelength value now to None to avoid possible mistakes or misuse
        modified_parameters._set_wavelength_m_None()

        return modified_parameters


class Propagate_Planes_Layer_Mono(tf.keras.layers.Layer):
    """Fourier optics-based field propagator instance (reuses prop_param configurations to define input and output
    grids and distances). Computes the output field(s) a fixed distance away from an initial plane, given a set of
    input field(s) of a single wavelength.

    Once initialized for a given geometry and grid, it may be called repeatedly. 'wavelength_m' must be defined and is
    expected to be the same for all fields. Use propagate_panes_broadband_layer for multi-wavelength simulations.
    - Plane seperation distance is defined by parameters["sensor_distance_m"], regardless of if the output is the sensor
         plane or an intermediate plane.
    - The input grid is defined by parameters["ms_samplesM"] and parameters["ms_dx_m"].
    - The output grid is defined by parameters["sensor_dx_m"] and parameters["sensor_pixel_number"].
    Thus, the input grid and output grid can be fully specified by the user.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
    """

    def __init__(self, parameters):
        """propagate_plane_layer initialization.

        Args:
            `parameters` (prop_param): Settings object defining field propagation details. Wavelength for calculation
                is set by parameters["wavelength_m"].

        Raises:
            KeyError: parameters object must have 'wavelength_m' defined.
            ValueError: The 'wavelength_m' value must be a single float.
        """
        super(Propagate_Planes_Layer_Mono, self).__init__()
        self.parameters = parameters
        check_single_wavelength_parameters(parameters)

        # Allow for application of aperture like with psf propagation case
        aperture_trans, _ = gen_aperture_disk(parameters)
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

    def __call__(self, inputs, batch_loop=False):
        """propagate_planes_layer call function. Computes the field amplitude and phase at a parallel plane a distance
        away from the initial plane, for a single wavelength.

        A batch of field profiles may be passed in and computed at once. For multiple-wavelengths, use
        propagate_planes_broadband_layer. The batch may often represent the field for different polarization states of
        light or more generally, may represent the fields from different devices.

        Args:
            `inputs` (list): Length two list containing field amplitude in first arg and field phase in second:
                o `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
                    (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
                o `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
                    (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
            `batch_loop` (boolean, defaults False): Flag whether to loop over profile_batches or compute at once

        Returns:
            `list`: List of field amplitude(s) in the first argument and phase(s) in the second arg at the output plane.
                The shape of each is given via (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
                or (batch_size, 1, sensor_pixel_number["r"]).
        """
        use_dtype = self.parameters["dtype"]
        field_amplitude = inputs[0]
        field_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(field_amplitude):
            field_amplitude = tf.convert_to_tensor(field_amplitude, dtype=use_dtype)
        else:
            if not field_amplitude.dtype == use_dtype:
                field_amplitude = tf.cast(field_amplitude, use_dtype)

        if not tf.is_tensor(field_phase):
            field_phase = tf.convert_to_tensor(field_phase, dtype=self.parameters["dtype"])
        else:
            if not field_phase.dtype == use_dtype:
                field_phase = tf.cast(field_phase, use_dtype)

        # Check if convenient input rank is used (Rank 4 with size 1 extra dimension)
        input_rank = tf.rank(field_amplitude)
        if input_rank == tf.TensorShape(4):
            field_amplitude = tf.squeeze(field_amplitude, 0)
            field_phase = tf.squeeze(field_phase, 0)

        # Apply the metasurface aperture
        field_amplitude = field_amplitude * self.aperture_trans

        if batch_loop:
            out = loopBatch_field_propagation(field_amplitude, field_phase, self.parameters)
        else:
            out = field_propagation(field_amplitude, field_phase, self.parameters)

        return out
