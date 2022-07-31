from .core.field_aperture import sqrt_energy_illumination, gen_aperture_disk
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf


def getCoordinates_vector(pixel_number_dict, dx_dict, radial_symmetry, dtype):
    if radial_symmetry:
        grid_x = tf.range(pixel_number_dict["r"], dtype=dtype)
        grid_y = tf.range(1, dtype=dtype)
    else:
        grid_x = tf.range(pixel_number_dict["x"], dtype=dtype)
        grid_y = tf.range(pixel_number_dict["y"], dtype=dtype)
        grid_x = grid_x - (len(grid_x) - 1) / 2
        grid_y = grid_y - (len(grid_y) - 1) / 2

    grid_x = grid_x * dx_dict["x"]
    grid_y = grid_y * dx_dict["y"]
    grid_x = tf.expand_dims(grid_x, 0)
    grid_y = tf.expand_dims(grid_y, 0)

    return grid_x, grid_y


def getCoordinates_mesh(pixel_number_dict, dx_dict, radial_symmetry, dtype):
    grid_x, grid_y = getCoordinates_vector(pixel_number_dict, dx_dict, radial_symmetry, dtype)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    return grid_x, grid_y


def focus_lens_init(parameters, wavelength_m_aslist, focal_distance_m_aslist, focus_offset_m_aslist):
    """Wrapper function for gen_focusing_profile. Generate a stack of theoretical transmittance and phase profiles for
    focusing light of particular depths and wavelengths, onto the sensor plane.

    This is sometimes more convenient to call than `gen_focusing_profiles` directly as it does not require explicitly
    unpacking the relevent function inputs in another script. Returns the profile(s) for focusing given the incident
    wavelength, object distance, of off-axis focal shift combinations passed in list form.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details. wavelength parameter key is not
            used. Only the geometry of the metasurface grid is pulled out.
        `wavelength_m_aslist` (list): N length list of floats defining incident wavelength_m in meters, to be designed
            for by the metasurface profile instantiation.
        `focal_distance_m_aslist` (list): N length list containing in-focus object-to-lens plane distances, in units of
            meters.
        `focus_offset_m_aslist` (list): N length list containing the off-axis focal-point shift at the sensor,
            via {"x": np.float, "y": np.float}.

    Raises:
        TypeError: focal_distance_m_aslist and focus_offset_m_aslist must be passed as list
        ValueError: All inputted lists must have same number of items
        ValueError: lens radius exceeds the minimum radius of the initialzed lens-grid

    Returns:
        `np.float`: Transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Phase profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Aperture transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or (n, 1, ms_samplesM["r"]).
        `np.float`: Sqrt of total energy incident on the metasurface, post-aperture, of shape (n, 1, 1).

    """

    # Unpack necessary arguments from parameters
    sensor_distance_m = parameters["sensor_distance_m"]
    ms_dx_m = parameters["ms_dx_m"]
    ms_samplesM = parameters["ms_samplesM"]
    radial_symmetry = parameters["radial_symmetry"]
    radius_m = parameters["radius_m"]

    # Handle exception if focus_distance, wavelength, or offset are not passed as lists
    if (
        (type(focal_distance_m_aslist) is not list)
        or (type(focus_offset_m_aslist) is not list)
        or (type(wavelength_m_aslist) is not list)
    ):
        raise TypeError("focus_lens_init: focal_distance_m_aslist and focus_offset_m_aslist must be passed as list")

    # Handle exception if list lengths are not all the same
    length = len(wavelength_m_aslist)
    if not all(len(lst) == length for lst in [focal_distance_m_aslist, focus_offset_m_aslist]):
        raise ValueError("focus_lens_init: All inputted lists must have the same length")

    # Handle exception if focusing lens radius exceeds the radius of the unpadded lens space
    maxradx = ms_dx_m["x"] * ((ms_samplesM["x"] - 1) / 2)
    maxrady = ms_dx_m["y"] * ((ms_samplesM["y"] - 1) / 2)
    if radius_m:
        if radius_m > np.min([maxradx, maxrady]):
            raise ValueError("focus_lens_init: lens radius exceeds the minimum radius of the initialzed lens-grid")

    # Loop over focusing lenses and generate metasurfaces profiles
    lens_transmittanceStack = []
    lens_phaseStack = []
    aperture_transmittanceStack = []
    for iter in range(len(focal_distance_m_aslist)):
        lens_transmittance, lens_phase, aperture_transmittance, _ = gen_focusing_profile(
            ms_samplesM,
            ms_dx_m,
            wavelength_m_aslist[iter],
            focal_distance_m_aslist[iter],
            focus_offset_m_aslist[iter],
            sensor_distance_m,
            radius_m,
            radial_symmetry,
        )

        complexLens = lens_transmittance * np.exp(1j * lens_phase)
        lens_transmittanceStack.append(np.abs(complexLens))
        lens_phaseStack.append(np.angle(complexLens))
        aperture_transmittanceStack.append(aperture_transmittance)

    lens_transmittanceStack = np.stack(lens_transmittanceStack)
    lens_phaseStack = np.stack(lens_phaseStack)
    aperture_transmittanceStack = np.stack(aperture_transmittanceStack)
    sqrt_energy_illum = sqrt_energy_illumination(aperture_transmittanceStack, ms_dx_m, radial_symmetry)

    return (
        lens_transmittanceStack,
        lens_phaseStack,
        aperture_transmittanceStack,
        sqrt_energy_illum,
    )


def gen_focusing_profile(
    ms_samplesM, ms_dx_m, wavelength_m, focal_distance_m, focus_offset_m, sensor_distance_m, radius_m, radial_symmetry
):
    """Generate the metasurface phase and transmittance profile for ideal focusing of light. Focusing is designed for a
    single, input wavelength, object plane distance/depth, and sensor distance. The lens may also be specified to focus
    light with an off-axis focal shift.

    Args:
        `ms_samplesM` (dict): Dictionary specifying the number of points in the metasurface cartesian grid, of form
            {"x": float, "y": float, "r": float}.
        `ms_dx_m` (dict): Dictionary specifying the grid discretization/pitch along x and y, of form {"x": float, "y": float}.
        `wavelength_m` (float): Wavelength of incident light, in units of m.
        `focal_distance_m` (float): Object plane distance/depth that will be imaged in focus, in units of m.
        `focus_offset_m` (dict): Off-axis shift of the focused psf at the sensor plane, in units of m and of the form
            {"x": float, "y": float}.
        `sensor_distance_m` (float): Distance from the metasurface to the sensor plane, in units of m.
        `radius_m` (float): Radius of the metasurface via a circular field aperture placed before the metasurface. If
            None, no field aperture is considered and a square metasurface is returned.
        `radial_symmetry` (bool): Radial symmetry flag denoting if the profile to be returned is 2D or a 1D radial
            vector of phase and transmission.

    Returns:
        `float`: Metasurface transmittance, of shape (ms_samplesM["y"], ms_samplesM["x"]) or (1, ms_samplesM["r"])
        `float`: Metasurface phase profile, of shape (ms_samplesM["y"], ms_samplesM["x"]) or (1, ms_samplesM["r"])
        `float`: Pre-metasurface field aperture (None or circular disk), of shape (ms_samplesM["y"], ms_samplesM["x"])
            or (1, ms_samplesM["r"])
        `float`: Total energy passing through the aperture and incident on the metasurface, used for normalizing psfs downstream.
    """
    # Generate coordinate grid on the lens plane
    xx, yy = np.meshgrid(np.arange(ms_samplesM["x"]), np.arange(ms_samplesM["y"]))
    xx = xx - (xx.shape[1] - 1) / 2
    yy = yy - (yy.shape[0] - 1) / 2
    xx = xx * ms_dx_m["x"]
    yy = yy * ms_dx_m["y"]

    # Define focusing profile phase
    lens_phase = (
        -2
        * np.pi
        / wavelength_m
        * (
            np.sqrt(focal_distance_m**2 + xx**2 + yy**2)
            + np.sqrt(sensor_distance_m**2 + (xx - focus_offset_m["x"]) ** 2 + (yy - focus_offset_m["y"]) ** 2)
        )
    )

    # Define transmittance of lens space
    lens_transmittance = np.ones_like(lens_phase)

    # Wrap to 2pi
    complex_lens = lens_transmittance * np.exp(1j * lens_phase)
    lens_phase = np.angle(complex_lens)
    # lens_transmittance = np.abs(lens_transmittance)

    # Define the lens aperture
    # Add a small transmittance to the block portion to avoid nan gradients downstream!
    aperture_transmittance = np.ones_like(lens_phase)
    if radius_m:
        flg = ((np.sqrt(xx**2 + yy**2) <= radius_m)).astype(np.float32) + 1e-6
        aperture_transmittance *= flg

    # Get measure of total radiance passed through the aperture onto the metasurface
    sqrt_energy_illum = sqrt_energy_illumination(np.expand_dims(aperture_transmittance, 0), ms_dx_m, False)

    # Handle return case if radial_symmetry flag is active
    # If the radius is of length N, the total size of the full 2D lens is (2N-1)x(2N-1)
    if radial_symmetry:
        ms_samplesM_r = ms_samplesM["r"]
        lens_transmittance = lens_transmittance[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]
        lens_phase = lens_phase[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]
        aperture_transmittance = aperture_transmittance[ms_samplesM_r - 1 : ms_samplesM_r, ms_samplesM_r - 1 :]

    return lens_transmittance, lens_phase, aperture_transmittance, sqrt_energy_illum


def randomPhase_lens_init(parameters, numlenses):
    """Generate a random phase profile with unity transmittance.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details including metasurface to sensor
            distance.
        `numlenses` (int): number n of random transmission and phase profiles to generate

    Returns:
        `np.float`: Unity transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Random phase profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Aperture transmittance profiles, of shape (n, ms_samplesM["y"], ms_samplesM["x"]) or
            (n, 1, ms_samplesM["r"]).
        `np.float`: Sqrt of total energy incident on the metasurface, post-aperture, of shape (n, 1, 1).
    """
    ms_samplesM = parameters["ms_samplesM"]
    radial_symmetry = parameters["radial_symmetry"]
    numpts_y = 1 if radial_symmetry else ms_samplesM["y"]
    numpts_x = ms_samplesM["r"] if radial_symmetry else ms_samplesM["x"]

    lens_trans = np.ones((numlenses, numpts_y, numpts_x))
    aperture, sqrt_energy_illum = gen_aperture_disk(parameters)
    lens_phase = 2 * np.pi * np.random.rand(lens_trans.shape)

    return lens_trans, lens_phase, aperture, sqrt_energy_illum


def gen_idmatrix_mask_seive(parameters, numSets):
    # create the fundamental tiled cell pattern
    # Allow for rectangular tile instead of square
    rootNumSets = np.sqrt(numSets)
    if np.round(rootNumSets) == rootNumSets:
        unitCellLenx = int(rootNumSets)
        unitCellLeny = int(rootNumSets)
    elif np.round(rootNumSets) > rootNumSets:
        unitCellLenx = int(np.ceil(rootNumSets))
        unitCellLeny = unitCellLenx
    else:
        unitCellLenx = int(np.ceil(rootNumSets))
        unitCellLeny = int(np.floor(rootNumSets))

    numInCell = unitCellLenx * unitCellLeny
    baseUnit = np.arange(1, numSets + 1, 1)
    baseVector = baseUnit
    while len(baseVector) < numInCell:
        baseVector = np.concatenate((baseVector, baseUnit))
    baseVector = baseVector.flatten()

    baseVector = np.reshape(baseVector[0:numInCell], (unitCellLenx, unitCellLeny))

    # Tile the repeating unit
    ms_samplesM = parameters["ms_samplesM"]
    numRepsx = int(ms_samplesM["x"] / unitCellLenx + 1)
    numRepsy = int(ms_samplesM["y"] / unitCellLeny + 1)
    tileMat = np.tile(baseVector, (numRepsx, numRepsy))

    # Cut excess mask that may result by overhanging tile
    tileMat = tileMat[0 : ms_samplesM["y"], 0 : ms_samplesM["x"]]

    return tileMat


def mutliplexing_mask_seive(parameters, numSets):
    """Generate a stack of checkerboard-like, binary masks, useful for designing spatially multiplexed metasurfaces.

    Note: This function produces an ideal interleave behavior (equal energy across masks) only for values of numSets
    that are perfect squares, e.g. 4, 9. In other cases, it is better to use random, orthogonal binary masks.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details including metasurface to sensor distance.
        `numSets` (int): number of orthogonal masks to generate.

    Raises:
        ValueError: cant do multiplexing with radial symmetry flag
        TypeError: numSets must be an integer
        ValueError: numSets should be greater than 1

    Returns:
        `np.float`: Set of seive binary masks, of shape (numSets, ms_samplesM["y"], ms_samplesM["x"]).
    """
    if parameters["radial_symmetry"]:
        raise ValueError("mutliplexing_mask_seive: cant do multiplexing with radial symmetry flag")

    if not isinstance(numSets, int):
        raise TypeError("gen_idmatrix_mask_seive: numSets must be an integer")

    if numSets <= 1:
        raise ValueError("gen_multiplexing_mask: numSets should be greater than 1")

    if not (int(np.sqrt(numSets) + 0.5) ** 2 == numSets):
        print(
            "Warning: numSets produce ideal sampling behavior only for perfect square value; You should use random multiplex instead"
        )

    idMatrix = gen_idmatrix_mask_seive(parameters, numSets)
    maskStack = []
    for i in range(numSets):
        thisMask = np.copy(idMatrix)
        thisMask[np.where(thisMask != i + 1)] = 0
        thisMask[np.where(thisMask == i + 1)] = 1
        maskStack.append(thisMask)

    return np.stack(maskStack)


def gen_idmatrix_randomCodedAperture(parameters, numSets, featuresize):
    # Feature size is a dict, "x" and "y", pixel pitch for the mask
    # e.g. is binary, transmissive slm masks which will certainly have different
    # feature sizes vs the metasurface unit cell seize
    ms_dx_m = parameters["ms_dx_m"]
    ms_samplesM = parameters["ms_samplesM"]
    ratiox = featuresize["x"] / ms_dx_m["x"]
    ratioy = featuresize["y"] / ms_dx_m["y"]

    downsampledShape = (
        int(ms_samplesM["y"] / ratioy + 1),
        int(ms_samplesM["x"] / ratiox + 1),
    )
    idmatrix = np.random.randint(1, numSets + 1, size=downsampledShape)
    # Use zoom to upsample
    idmatrix = zoom(idmatrix, (ratiox, ratioy), order=0)
    idmatrix = idmatrix[: ms_samplesM["y"], : ms_samplesM["x"]]

    return idmatrix


def codedapertures_binaryRandom(parameters, numSets, featuresize):
    """Generates a set of orthogonal, random coded binary masks ontop the metasurface grid. The feature size on the
    mask may be larger than the metasurface grid discretization.

    Args:
        `parameters` (prop_param):  Settings object defining field propagation details including metasurface to sensor distance.
        `numSets` (int): number of orthogonal masks to generate.
        `featuresize` (dict): Smallest binary segment of the coded aperture, defined via the dict {"x": np.float, "y": np.float}.

    Raises:
        ValueError: Radial_symmetry flag must be False.
        TypeError: numSets must be an integer.
        ValueError: numSets must be >1.
        ValueError: Feature sizes on the coded aperture must not be smaller than the metasurface sampling specified in parameters.
        ValueError: Feature sizes must be an integer multiple of the metasurface pixel size.

    Returns:
        `np.float`: Binary mask stack, of shape, (numSets, ms_samplesM["y"], ms_samplesM["x"]).
    """

    # Run checks on inputs
    if parameters["radial_symmetry"]:
        raise ValueError("codedaperture_binaryRandom: cant do multiplexing with radial symmetry flag")

    if not isinstance(numSets, int):
        raise TypeError("codedaperture_binaryRandom: n must be an integer")

    if numSets <= 1:
        raise ValueError("codedaperture_binaryRandom: numSets should be greater than 1")

    # Verify that the feature size requested is not smaller than the metasurface sampling
    # This may be changed later but for now, also ensure that featuresize is an int multiple of ms pixel size
    ms_dx_m = parameters["ms_dx_m"]
    if (ms_dx_m["x"] > featuresize["x"]) or (ms_dx_m["y"] > featuresize["y"]):
        raise ValueError(
            "gen_idmatrix_randomCodedAperture: featursize of mask cannot have smaller sampling than the metasurface"
        )

    if (np.mod(featuresize["x"] / ms_dx_m["x"], 1) != 0) or (np.mod(featuresize["y"] / ms_dx_m["y"], 1) != 0):
        raise ValueError(
            "gen_idmatrix_randomCodedAperture: featursize of mask should be integer multiple of pixel pitch"
        )

    idmatrix = gen_idmatrix_randomCodedAperture(parameters, numSets, featuresize)
    maskStack = []
    for i in range(numSets):
        thisMask = np.copy(idmatrix)
        thisMask[np.where(thisMask != i + 1)] = 0
        thisMask[np.where(thisMask == i + 1)] = 1
        maskStack.append(thisMask)

    return np.stack(maskStack)
