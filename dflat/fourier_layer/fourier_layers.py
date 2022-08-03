import tensorflow as tf

from dflat.data_structure import prop_params
from .core.field_aperture import gen_aperture_disk
from .core.batched_FourierOpt import *


def check_single_wavelength_parameters(parameters):
    if not ("wavelength_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_m")

    if not (type(parameters["wavelength_m"]) == float):
        raise ValueError("Wavelength should be a single float")
    return


def check_broadband_wavelength_parameters(parameters):
    if not ("wavelength_set_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_set_m")
    return


class PSF_Layer_Mono(tf.keras.layers.Layer):
    """Fourier optics-based, point-spread function computing instance (single prop_param setting configuration and
    single wavelength). Computes the psf(s) of the optical system for different point-source(s), given metasurface
    modulation profile(s)--transmittance and phase--on a single wavelength.

    Once initialized for a given geometry and grid, it may be called repeatedly. For psfs as a function of many
    wavelengths, use psf_broadband_layer instead.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `aperture_trans` (tf.float64): Pre-metasurface field aperture used in calculation, of shape
            (1, ms_samplesM["y"], ms_samplesM["x"]).
    """

    def __init__(self, parameters):
        """Fourier PSF Layer Initialization.

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

    def __call__(self, inputs, point_source_locs):
        """The psf_layer call function. Computes the PSF, given a set of point_source_locs and the set of phase and
        transmittance profiles.

        For a polarization-sensitive metasurface, a stacked pair of phase and transmittance profiles may be passed in
        at once, corresponding to the optical response on orthogonal, polarization basis states. The metasurface
        batch dimension may be more generally used to represent the phase and transmittance modulation for a set of many
        polarization states or to represent a set of many different metasurfaces.

        Args:
            `ms_trans` (float): Transmittance profile of the optical metasurface, of shape
                (1 or 0, profile_batch, ms_samplesM['y'], ms_samplesM['x']) or (1 or 0, profile_batch, 1, ms_samplesM['r']). Note
                that the first axis is optional.
            `ms_phase` (float): Phase profile of the optical metasurface, of shape
                (1 or 0, profile_batch, ms_samplesM['y'], ms_samplesM['x']) or (1 or 0, 1, profile_batch, 1, ms_samplesM['r']). Note
                that the first axis is optional.
            `point_source_locs` (float): Tensor of point-source coordinates, of shape `(N,3)`.

        Returns:
            `list`: List containing the detector measured PSF intensity in the first argument and the phase in the
                second argument, of shape
                (1, profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        """
        ms_trans = inputs[0]
        ms_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(ms_trans):
            ms_trans = tf.convert_to_tensor(ms_trans, dtype=self.parameters["dtype"])

        if not tf.is_tensor(ms_phase):
            ms_phase = tf.convert_to_tensor(ms_phase, dtype=self.parameters["dtype"])

        if not tf.is_tensor(point_source_locs):
            point_source_locs = tf.convert_to_tensor(point_source_locs, dtype=self.parameters["dtype"])

        # Check if convenient input rank is used
        ms_rank = tf.shape(ms_trans).shape
        if ms_rank == 4:
            ms_trans = tf.squeeze(ms_trans, 0)
            ms_phase = tf.squeeze(ms_phase, 0)

        # Apply the metasurface aperture
        ms_trans = ms_trans * self.aperture_trans

        return batched_psf_measured(ms_trans, ms_phase, self.__sqrt_energy_illum, point_source_locs, self.parameters)


class PSF_Layer(tf.keras.layers.Layer):
    """Fourier optics-based, point-spread function computing instance (single prop_param setting configuration)
    Computes the psf of the optical system for multiple wavelengths given a set of metasurface modulation profile(s)
    for each wavelength and a set of object distances.

    Once initialized for a given geometry and grid, it may be called repeatedly. 'wavelength_set_m' must be defined.

    Attributes:
        `parameters` (prop_params): Single settings object used during initialization of propagator.
        `parameters_list` (list of prop_params objects): A list of prop_param configuration objects
            initialized for each wavelength in the set.
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

        # Generate the Fourier grids for each wavelength
        self.parameters_list = self.__generate_simParam_set()

        aperture_trans, sqrt_energy_illum = gen_aperture_disk(parameters)
        self.__sqrt_energy_illum = tf.convert_to_tensor(sqrt_energy_illum, dtype=parameters["dtype"])
        self.aperture_trans = tf.convert_to_tensor(aperture_trans, dtype=parameters["dtype"])

    def __call__(self, inputs, point_source_locs):
        """The psf_broadband_layer call function. Computes the PSF, given a set of point_source_locs and a set of phase
        and transmittance profiles for each wavelength in the set. This call enables overloading, such that the
        metasurface profiles may be uniquely defined for each wavelength or assumed to be the same across wavelength
        channels.

        For a polarization-sensitive metasurface, a stacked pair of phase and transmittance profiles may be passed in
        at once, corresponding to the optical response on two orthogonal, polarization basis states for each wavelength
        channel in the set. The metasurface batch dimension may be more generally used to represent the phase and
        transmittance for a general set of many polarization states or to represent a set of many different
        metasurfaces at each wavelength.

        Args:
            `ms_trans` (float): Transmittance profile(s) of the optical metasurface(s), of shape
                (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']),
                or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                same across wavelength, one may pass in transmittance of shape,
                (profile_batch, ms_samplesM['y'], ms_samplesM['x']) or  (profile_batch, 1, ms_samplesM['r']).
            `ms_phase` (float): Phase profile(s) of the optical metasurface(s), of shape
                (len(wavelength_set_m), profile_batch, ms_samplesM['y'], ms_samplesM['x']).
                or (len(wavelength_set_m), profile_batch, 1, ms_samplesM['r']). Alternatively, if the profiles are the
                same across wavelength, one may pass in phase of shape, (profile_batch, ms_samplesM['y'], ms_samplesM['x'])
                or (profile_batch, 1, ms_samplesM['r']).
            `point_source_locs` (float): Tensor of point-source coordinates, of shape (N,3).

        Returns:
            `list`: List containing the detector measured PSF intensity in the first argument and the phase in the
                second argument, of shape
                (len(wavelength_set_m), profile_batch, num_point_sources, sensor_pixel_number["y"], sensor_pixel_number["x"])
        """
        dtype = self.parameters["dtype"]
        ms_trans = tf.cast(inputs[0], dtype)
        ms_phase = tf.cast(inputs[1], dtype)

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(ms_trans):
            ms_trans = tf.convert_to_tensor(ms_trans, dtype=self.parameters["dtype"])

        if not tf.is_tensor(ms_phase):
            ms_phase = tf.convert_to_tensor(ms_phase, dtype=self.parameters["dtype"])

        if not tf.is_tensor(point_source_locs):
            point_source_locs = tf.convert_to_tensor(point_source_locs, dtype=self.parameters["dtype"])

        # Apply the metasurface aperture
        ms_rank = tf.shape(ms_trans).shape
        if ms_rank == 3:
            ms_trans = ms_trans * self.aperture_trans
        elif ms_rank == 4:
            ms_trans = ms_trans * tf.expand_dims(self.aperture_trans, 0)

        return broadband_batched_psf_measured(
            ms_trans, ms_phase, self.__sqrt_energy_illum, point_source_locs, self.parameters_list
        )

    def __generate_simParam_set(self):
        wavelength_set_m = self.parameters["wavelength_set_m"]
        parameters_list = []

        for wavelength in wavelength_set_m:
            setting_dict = self.parameters.get_dict()
            del setting_dict["wavelength_set_m"]
            setting_dict["wavelength_m"] = wavelength
            parameters_list.append(prop_params(setting_dict))

        return parameters_list


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

    def __call__(self, inputs):
        """propagate_planes_layer call function. Computes the field amplitude and phase at a parallel plane a distance
        away from the initial plane, for a single wavelength.

        A batch of field profiles may be passed in and computed at once. For multiple-wavelengths, use
        propagate_planes_broadband_layer. The batch may often represent the field for different polarization states of
        light or more generally, may represent the fields from different devices.

        Args:
            `field_amplitude` (tf.float64): Amplitude(s) at the initial plane, in shape of
                (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])
            `field_phase` (tf.float64): Phase(s) at the initial plane, in shape of
                (batch_size, ms_samplesM["y"], ms_samplesM["x"]) or (batch_size, 1, ms_samplesM["r"])

        Returns:
            `list`: List of field amplitude(s) in the first argument and phase(s) in the second arg at the output plane.
                The shape of each is given via (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
                or (batch_size, 1, sensor_pixel_number["r"]).
        """
        field_amplitude = inputs[0]
        field_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(field_amplitude):
            field_amplitude = tf.convert_to_tensor(field_amplitude, dtype=self.parameters["dtype"])
        if not tf.is_tensor(field_phase):
            field_phase = tf.convert_to_tensor(field_phase, dtype=self.parameters["dtype"])

        return field_propagation(field_amplitude, field_phase, self.parameters)


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

    def __call__(self, inputs):
        """propagate_planes_broadband_layer call function. Computes the field amplitude and phase at a parallel plane a
        distance away from the initial plane, for multiple wavelength channels.

        A batch of field profiles (optionally, a batch specified for each wavelength channel) may be passed in and
        computed at once. The field profile batches may often represent the field for different polarization states of
        light or more generally, may represent the fields from different devices, at each wavelength.

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

        Returns:
            `list`: List of field amplitude(s) in the first argument and phase(s) in the second arg at the output plane.
            The shape of each is given via (len(wavelength_set_m), num_profiles, sensor_pixel_number["y"], sensor_pixel_number["x"])
            or (len(wavelength_set_m), num_profiles, 1, sensor_pixel_number["r"]).
        """
        field_amplitude = inputs[0]
        field_phase = inputs[1]

        # Allow for tensor conversion when non-tensor is passed as input
        if not tf.is_tensor(field_amplitude):
            field_amplitude = tf.convert_to_tensor(field_amplitude, dtype=self.parameters["dtype"])
        if not tf.is_tensor(field_phase):
            field_phase = tf.convert_to_tensor(field_phase, dtype=self.parameters["dtype"])

        return broadband_batched_propagation(field_amplitude, field_phase, self.parameters_list)

    def __generate_simParam_set(self):
        wavelength_set_m = self.parameters["wavelength_set_m"]
        parameters_list = []

        for wavelength in wavelength_set_m:
            setting_dict = self.parameters.get_dict()
            del setting_dict["wavelength_set_m"]
            setting_dict["wavelength_m"] = wavelength
            parameters_list.append(prop_params(setting_dict))

        return parameters_list
