import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .hankel import radial_crop_or_pad


def batch_interp_regular_nd(real_refdat_2d, ref_grid_min, ref_grid_max, x_grid_mesh, y_grid_mesh):
    """_summary_

    Args:
        real_refdat_2d (tf.float): nD, real tensor corresponding to a regular grid defined by ref limits. 
        ref_grid_min (list): n element list specifying the min of the reference grid along each dimension.
        ref_grid_max (_type_): n element list specifying the max of the reference grid along each dimension
        x_grid_mesh (_type_): _description_
        y_grid_mesh (_type_): _description_
        parameters (_type_): _description_
    """
    x = tf.stack([tf.reshape(x_grid_mesh, -1), tf.reshape(y_grid_mesh, -1)], axis=1)
    interp_dat = tfp.math.batch_interp_regular_nd_grid(x, ref_grid_min, ref_grid_max, real_refdat_2d, axis=-2)

    return tf.reshape(interp_dat, shape=(real_refdat_2d.shape[0], x_grid_mesh.shape[1], x_grid_mesh.shape[0]))


def resize_area(image, input_size, resize_ratio):
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
    # Pad the image with zero if needed
    image = tf.pad(
        image,
        [
            [0, 0],
            [0, np.remainder(input_size["y"], resize_ratio["y"])],
            [0, np.remainder(input_size["x"], resize_ratio["x"])],
            [0, 0],
        ],
        mode="CONSTANT",
    )

    # Define the filter which requires shape [filter_height, filter_width, in_channels, out_channels]
    # Use a simple box filter
    rectFilter = tf.expand_dims(
        tf.expand_dims(tf.ones([resize_ratio["y"], resize_ratio["x"]], dtype=image.dtype), -1), -1
    )

    # convolve the image with the filter using stride defined by ratio
    outimage = tf.nn.conv2d(image, rectFilter, strides=[1, resize_ratio["y"], resize_ratio["x"], 1], padding="VALID")

    return outimage


def resample_intensity_sensor(intensity, input_size, resize_ratio):
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
    return resize_area(intensity, input_size, resize_ratio)


def resample_phase_sensor(phase, input_size, resize_ratio):
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
    phasex = resize_area(tf.math.cos(phase), input_size, resize_ratio) / resize_ratio["x"] / resize_ratio["y"]
    phasey = resize_area(tf.math.sin(phase), input_size, resize_ratio) / resize_ratio["x"] / resize_ratio["y"]

    return tf.math.atan2(phasey, phasex)


def sensorMeasurement_intensity_phase(sensor_intensity, sensor_phase, parameters):
    """Returns both the measured intensity on the detector and the averaged phase on the detector pixels, given the 
    intensity and phase on a grid just above the detector face.
    
    Note: In the current version, the measurement implementation does NOT enable fractional resize treatment. As an
    important consquence, the integer rounded resize will be slightly incorrect. 

    Args:
        `sensor_intensity` (tf.float64): Field intensity at the sensor plane, of shape (batch_size, calc_samplesN["y"], calc_samplesN["x"]).
        `sensor_phase` (tf.float64): Field phase at the sensor plane, of shape (batch_size, calc_samplesN["y"], calc_samplesN["x"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Intensity measured on the detector pixel array, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
        `tf.float64`: Average phase measured on the detector pixel array, of shape (batch_size, sensor_pixel_number["y"], sensor_pixel_number["x"])
    """

    ### Our manual conv2D area-resize requires integer resize ratios so first we reinterpolate the sensor grid measurements
    # Unpack parameters
    calc_samplesN = parameters["calc_samplesN"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]

    # Define the sensor plane grid (non-centered)
    max_grid_span_x = calc_samplesN["x"] * calc_sensor_dx_m["x"]
    max_grid_span_y = calc_samplesN["y"] * calc_sensor_dx_m["y"]
    sens_grid_min = [0.0, 0.0]
    sens_grid_max = [max_grid_span_x, max_grid_span_y]

    round_ratio_pixel_grid = {
        "x": int(np.round(sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"])),
        "y": int(np.round(sensor_pixel_size_m["y"] / calc_sensor_dx_m["y"])),
    }
    interp_sensor_dx = {
        "x": sensor_pixel_size_m["x"] / round_ratio_pixel_grid["x"],
        "y": sensor_pixel_size_m["y"] / round_ratio_pixel_grid["y"],
    }

    # Call interp on new sensor plane grid
    if parameters["accurate_measurement"]:
        interp_grid_x, interp_grid_y = tf.meshgrid(
            tf.range(0, max_grid_span_x, delta=interp_sensor_dx["x"], dtype=dtype),
            tf.range(0, max_grid_span_y, delta=interp_sensor_dx["y"], dtype=dtype),
        )
        sensor_intensity = batch_interp_regular_nd(
            sensor_intensity, sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y
        )
        sensor_phase = batch_interp_regular_nd(
            sensor_phase, sens_grid_min, sens_grid_max, interp_grid_x, interp_grid_y
        )
    new_size = {"x": sensor_intensity.shape[2], "y": sensor_intensity.shape[0]}

    # Now call the resize
    sensor_intensity = resample_intensity_sensor(
        tf.expand_dims(sensor_intensity, -1), new_size, round_ratio_pixel_grid
    )
    sensor_intensity = tf.image.resize_with_crop_or_pad(
        sensor_intensity, sensor_pixel_number["y"], sensor_pixel_number["x"]
    )

    sensor_phase = resample_phase_sensor(tf.expand_dims(sensor_phase, -1), new_size, round_ratio_pixel_grid)
    sensor_phase = tf.image.resize_with_crop_or_pad(sensor_phase, sensor_pixel_number["y"], sensor_pixel_number["x"])

    return tf.squeeze(sensor_intensity, -1), tf.squeeze(sensor_phase, -1)


def sensorMeasurement_intensity_phase_radialData(sensor_intensity, sensor_phase, parameters):
    """Returns both the measured intensity on the detector and the averaged phase on the detector pixels, given the
    intensity and phase on a grid just above the detector face.

    Note: In the current version, the measurement implementation does NOT enable fractional resize treatment. As an
    important consquence, the integer rounded resize will be slightly incorrect.

    Args:
        `sensor_intensity` (tf.float64): Field intensity at the sensor plane, of shape (batch_size, 1, calc_samplesN["r"]).
        `sensor_phase` (tf.float64): Field phase at the sensor plane, of shape (batch_size, 1, calc_samplesN["r"]).
        `parameters` (prop_params): Settings object defining field propagation details.

    Returns:
        `tf.float64`: Intensity measured on the detector pixel array, of shape (batch_size, 1, sensor_pixel_number["r"])
        `tf.float64`: Average phase measured on the detector pixel array, of shape (batch_size, 1, sensor_pixel_number["r"])
    """

    ### Our manual conv2D area-resize requires integer resize ratios so first we reinterpolate the sensor grid measurements
    # Unpack parameters
    calc_samplesN = parameters["calc_samplesN"]
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    dtype = parameters["dtype"]

    # Define the sensor plane grid
    sens_grid_max = calc_samplesN["r"] * calc_sensor_dx_m["x"]
    round_ratio_pixel_grid_x = int(round(sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"]))
    interp_sensor_dx = sensor_pixel_size_m["x"] / round_ratio_pixel_grid_x

    # Call interp on new sensor plane grid
    if parameters["accurate_measurement"]:
        r = tf.range(0, sens_grid_max, delta=interp_sensor_dx, dtype=dtype)
        sensor_intensity = tfp.math.interp_regular_1d_grid(r, 0.0, sens_grid_max, sensor_intensity)
        sensor_phase = tfp.math.interp_regular_1d_grid(r, 0.0, sens_grid_max, sensor_phase)

    new_size = {"x": sensor_intensity.shape[2], "y": sensor_intensity.shape[0]}
    sensor_intensity = resample_intensity_sensor(
        tf.expand_dims(sensor_intensity, -1), new_size, {"x": round_ratio_pixel_grid_x, "y": 1}
    )
    sensor_phase = resample_intensity_sensor(
        tf.expand_dims(sensor_phase, -1), new_size, {"x": round_ratio_pixel_grid_x, "y": 1}
    )

    # Resize or crop the signal
    sensor_intensity = radial_crop_or_pad(tf.squeeze(sensor_intensity, -1), {"r": sensor_pixel_number["r"]})
    sensor_phase = radial_crop_or_pad(tf.squeeze(sensor_phase, -1), {"r": sensor_pixel_number["r"]})

    return sensor_intensity, sensor_phase
