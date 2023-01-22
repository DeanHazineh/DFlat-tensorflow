import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .ops_hankel import radial_crop_or_pad


def batch_interp_regular_nd(real_refdat_2d, ref_grid_min, ref_grid_max, x_grid_mesh, y_grid_mesh):
    """Helper function when calling batch_interp_regular_nd_grid.

    Args:
        real_refdat_2d (tf.float): Real-valued tensor corresponding to reference ydata on ref_grid_min, ref_grid_max of
            shape (batch_size, Ny, Nx).
        ref_grid_min (list): n=2 element list specifying the min of the reference grid along each dimension.
        ref_grid_max (_type_): n=2 element list specifying the max of the reference grid along each dimension
        x_grid_mesh (_type_): Rank n=2 tensor corresponding to the x coordinate of each pixel
        y_grid_mesh (_type_): Rank n=2 tensor corresponding to the y coordinate of each pixel
        parameters (_type_): _description_
    """

    # x is a list of coordinates to interpolate on of shape [..., D, nd] where nd is the space dimensionality and D
    # is the total number of points. This is to match the required input of the tensorflow call
    x = tf.stack([tf.reshape(y_grid_mesh, -1), tf.reshape(x_grid_mesh, -1)], axis=1)
    interp_dat = tfp.math.batch_interp_regular_nd_grid(x, ref_grid_min, ref_grid_max, real_refdat_2d, axis=-2)

    return tf.reshape(interp_dat, shape=(real_refdat_2d.shape[0], x_grid_mesh.shape[0], x_grid_mesh.shape[1]))


def resize_area(image, resize_ratio):
    """Autodifferentiable area resize implementation based on a strided convolution using tf.nn.conv2d.

    Currently, this implementation is only valid for integer resize, not fractional.

    Args:
        `image` (tf.dtype): Image to be resampled and box averaged, of shape (batch_size, Ny, Nx, channel_size)
        `input_size` (dict): Size of images, defined by dict {"x": float, "y": float}
        `resize_ratio` (dict): Resize factor along x and y, defined by dict {"x": float, "y": float}. Ratios are of
            input shape lengths to ouput shape lengths.

    Returns:
        `tf.float`: Area-resized image of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size)
    """
    # Define the filter which requires shape [filter_height, filter_width, in_channels, out_channels]
    # Use a simple box filter then do a 2D strided convolution with image
    rectFilter = tf.ones((resize_ratio["y"], resize_ratio["x"], 1, 1), dtype=image.dtype)
    outimage = tf.nn.conv2d(image, rectFilter, strides=[1, resize_ratio["y"], resize_ratio["x"], 1], padding="VALID")

    return outimage


def resample_intensity_sensor(intensity, resize_ratio):
    """Helper function calling the desired resize_area method to process the passed in intensity.

    Currently, only an autodifferentiable, non-fractional box-filter area-resize method is encoded and used.
    In the future, this function may be expanded to take other options.

    Args:
        `Intensity` (tf.float64): Field intensity to be resampled and box integrated, of shape (batch_size, Ny, Nx, channel_size)
        `Input_size` (dict): Size of images, defined by dict {"x": float, "y": float}
        `Resize_ratio` (dict): Resize factor along x and y, defined by dict {"x": float, "y": float}. Ratios are of
            input shape lengths to ouput shape lengths.

    Returns:
        `tf.float`: Area-resized intensity of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    return resize_area(intensity, resize_ratio)


def resample_phase_sensor(phase, resize_ratio):
    """Helper function calling the desired resize_area average method to process the passed in phase profile.

    Args:
        `Phase` (tf.float64): Field phase profile to be resampled and box averaged, of shape (batch_size, Ny, Nx, channel_size)
        `Input_size` (dict): Size of images, defined by dict {"x": float, "y": float}
        `Resize_ratio` (dict): Resize factor along x and y, defined by dict {"x": float, "y": float}. Ratios are of
            input shape lengths to ouput shape lengths.

    Returns:
        `tf.float`: Area-resized phase of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    # we want the output to be the average phase over the larger pixel rather than the sum
    phasex = resize_area(tf.math.cos(phase), resize_ratio) / resize_ratio["x"] / resize_ratio["y"]
    phasey = resize_area(tf.math.sin(phase), resize_ratio) / resize_ratio["x"] / resize_ratio["y"]

    return tf.math.atan2(phasey, phasex)


def sensorMeasurement_intensity_phase(sensor_intensity, sensor_phase, parameters):
    """Returns both the measured intensity on the detector and the averaged phase on the detector pixels, given the
    intensity and phase on a grid just above the detector face.

    Note: In the current version, the measurement implementation does NOT enable fractional resize treatment. As an
    important consquence, the integer rounded resize will be slightly incorrect.

    Args:
        `sensor_intensity` (tf.float64): Field intensity at the sensor plane, of shape (..., calc_samplesN["y"], calc_samplesN["x"]).
        `sensor_phase` (tf.float64): Field phase at the sensor plane, of shape (..., calc_samplesN["y"], calc_samplesN["x"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Intensity measured on the detector pixel array, of shape (..., sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float64`: Average phase measured on the detector pixel array, of shape (..., sensor_pixel_number["y"], sensor_pixel_number["x"])
    """
    if not (sensor_intensity.shape == sensor_phase.shape):
        raise ValueError("intensity and phase must be the same shape")

    ## Handle multi-dimension input
    input_rank = tf.shape(sensor_intensity).shape
    init_shape = sensor_intensity.shape
    if tf.math.equal(input_rank, tf.TensorShape(2)):
        sensor_intensity = tf.expand_dims(sensor_intensity, 0)
        sensor_phase = tf.expand_dims(sensor_phase, 0)

    if tf.math.greater(input_rank, tf.TensorShape(3)):
        sensor_intensity = tf.reshape(sensor_intensity, [-1, init_shape[-2], init_shape[-1]])
        sensor_phase = tf.reshape(sensor_phase, [-1, init_shape[-2], init_shape[-1]])

    ## Unpack parameters
    calc_samplesN = parameters["calc_samplesN"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]

    ### Only do convolutional resizing if the grid size and the sensor sizes are not the same (up to tolerance)
    tol = 0.1e-6
    if (abs(sensor_pixel_size_m["x"] - calc_sensor_dx_m["x"]) > tol) or (abs(sensor_pixel_size_m["y"] - calc_sensor_dx_m["y"]) > tol):
        ### Because we will do convolution-based resize, we take the calculated field and reinterpolate it onto a new grid
        # The sensor pixel size will be an integer multiple of the new grid enabling accurate convolution area sum

        round_ratio_pixel_grid = {
            "x": int(np.round(sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"])),
            "y": int(np.round(sensor_pixel_size_m["y"] / calc_sensor_dx_m["y"])),
        }

        if parameters["accurate_measurement"]:  # Eventually, change to always true and remove flag
            interp_sensor_dx = {
                "x": sensor_pixel_size_m["x"] / round_ratio_pixel_grid["x"],
                "y": sensor_pixel_size_m["y"] / round_ratio_pixel_grid["y"],
            }

            max_grid_span_x = calc_samplesN["x"] * calc_sensor_dx_m["x"]
            max_grid_span_y = calc_samplesN["y"] * calc_sensor_dx_m["y"]
            sens_grid_min = [0.0, 0.0]
            sens_grid_max = [max_grid_span_x, max_grid_span_y]

            interp_grid_x, interp_grid_y = tf.meshgrid(
                tf.range(0, max_grid_span_x, delta=interp_sensor_dx["x"], dtype=dtype),
                tf.range(0, max_grid_span_y, delta=interp_sensor_dx["y"], dtype=dtype),
            )

            sensor_intensity = batch_interp_regular_nd(sensor_intensity, sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y)
            # sensor_phase = batch_interp_regular_nd(sensor_phase, sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y)
            sensor_phase = tf.math.atan2(
                batch_interp_regular_nd(tf.math.sin(sensor_phase), sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y),
                batch_interp_regular_nd(tf.math.cos(sensor_phase), sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y),
            )

        # Now call the area resize methods for intensity and phase. We extend the dimension to add the required channel dimension
        sensor_intensity = tf.squeeze(resample_intensity_sensor(tf.expand_dims(sensor_intensity, -1), round_ratio_pixel_grid), -1)
        sensor_phase = tf.squeeze(resample_phase_sensor(tf.expand_dims(sensor_phase, -1), round_ratio_pixel_grid), -1)

    # Crop or pad with zeros to the sensor size
    sensor_intensity = tf.squeeze(
        tf.image.resize_with_crop_or_pad(tf.expand_dims(sensor_intensity, -1), sensor_pixel_number["y"], sensor_pixel_number["x"]), -1
    )
    sensor_phase = tf.squeeze(
        tf.image.resize_with_crop_or_pad(tf.expand_dims(sensor_phase, -1), sensor_pixel_number["y"], sensor_pixel_number["x"]), -1
    )

    # Return with the same batch_size shape
    new_shape = sensor_intensity.shape
    if input_rank != 3:
        sensor_intensity = tf.reshape(sensor_intensity, tf.concat([init_shape[:-2], new_shape[-2:]], axis=0))
        sensor_phase = tf.reshape(sensor_phase, tf.concat([init_shape[:-2], new_shape[-2:]], axis=0))

    return sensor_intensity, sensor_phase


def sensorMeasurement_intensity_phase_radialData(sensor_intensity, sensor_phase, parameters):
    """Returns both the measured intensity on the detector and the averaged phase on the detector pixels, given the
    intensity and phase on a grid just above the detector face.

    Note: In the current version, the measurement implementation does NOT enable fractional resize treatment. As an
    important consquence, the integer rounded resize will be slightly incorrect.

    Args:
        `sensor_intensity` (tf.float64): Field intensity at the sensor plane, of shape (..., 1, calc_samplesN["r"]).
        `sensor_phase` (tf.float64): Field phase at the sensor plane, of shape (..., 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Intensity measured on the detector pixel array, of shape (..., 1, sensor_pixel_number["r"])
        `tf.float64`: Average phase measured on the detector pixel array, of shape (..., 1, sensor_pixel_number["r"])
    """

    if not (sensor_intensity.shape == sensor_phase.shape):
        raise ValueError("intensity and phase must be the same shape")

    ## Handle multi-dimension input
    input_rank = tf.shape(sensor_intensity).shape
    init_shape = sensor_intensity.shape
    if tf.math.equal(input_rank, tf.TensorShape(2)):
        sensor_intensity = tf.expand_dims(sensor_intensity, 0)
        sensor_phase = tf.expand_dims(sensor_phase, 0)

    if tf.math.greater(input_rank, tf.TensorShape(3)):
        sensor_intensity = tf.reshape(sensor_intensity, [-1, init_shape[-2], init_shape[-1]])
        sensor_phase = tf.reshape(sensor_phase, [-1, init_shape[-2], init_shape[-1]])

    ## Unpack parameters
    calc_samplesN = parameters["calc_samplesN"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]

    ### Only do convolutional resizing if the grid size and the sensor sizes are not the same (up to tolerance)
    tol = 0.1e-6
    if (abs(sensor_pixel_size_m["x"] - calc_sensor_dx_m["x"]) > tol) or (abs(sensor_pixel_size_m["y"] - calc_sensor_dx_m["y"]) > tol):
        ### Because we will do convolution-based resize, we take the calculated field and reinterpolate it onto a new grid
        # The sensor pixel size will be an integer multiple of the new grid enabling accurate convolution area sum
        round_ratio_pixel_grid_x = int(round(sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"]))

        if parameters["accurate_measurement"]:  # Eventually, change to always true and remove flag
            # Define the sensor plane grid
            sens_grid_max = calc_samplesN["r"] * calc_sensor_dx_m["x"]
            interp_sensor_dx = sensor_pixel_size_m["x"] / round_ratio_pixel_grid_x

            r = tf.range(0, sens_grid_max, delta=interp_sensor_dx, dtype=dtype)
            sensor_intensity = tfp.math.interp_regular_1d_grid(r, 0.0, sens_grid_max, sensor_intensity)
            sensor_phase = tf.math.atan2(
                tfp.math.interp_regular_1d_grid(r, 0.0, sens_grid_max, tf.math.sin(sensor_phase)),
                tfp.math.interp_regular_1d_grid(r, 0.0, sens_grid_max, tf.math.cos(sensor_phase)),
            )

        # Now call the area resize methods for intensity and phase. We extend the dimension to add the required channel dimension
        sensor_intensity = tf.squeeze(
            resample_intensity_sensor(tf.expand_dims(sensor_intensity, -1), {"x": round_ratio_pixel_grid_x, "y": 1}), -1
        )
        sensor_phase = tf.squeeze(resample_phase_sensor(tf.expand_dims(sensor_phase, -1), {"x": round_ratio_pixel_grid_x, "y": 1}), -1)

    # Resize or crop the signal
    sensor_intensity = radial_crop_or_pad(sensor_intensity, {"r": sensor_pixel_number["r"]})
    sensor_phase = radial_crop_or_pad(sensor_phase, {"r": sensor_pixel_number["r"]})

    # Return with the same batch_size shape
    new_shape = sensor_intensity.shape
    if input_rank != tf.TensorShape(3):
        sensor_intensity = tf.reshape(sensor_intensity, tf.concat([init_shape[:-2], new_shape[-2:]], axis=0))
        sensor_phase = tf.reshape(sensor_phase, tf.concat([init_shape[:-2], new_shape[-2:]], axis=0))

    return sensor_intensity, sensor_phase
