import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from .ops_hankel import radial_conditional_resize_with_crop_or_pad


def resize_area(image, resize_ratio):
    """Autodifferentiable area resize implementation based on a strided convolution using tf.nn.conv2d.

    Currently, this implementation is only valid for integer resize, not fractional.

    Args:
        `image` (tf.dtype): Image to be resampled and box averaged, of shape (batch_size, Ny, Nx, channel_size)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)
    Returns:
        `tf.float`: Area-resized image of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size)
    """
    # Define the filter which requires shape [filter_height, filter_width, in_channels, out_channels]
    # Use a simple box filter then do a 2D strided convolution with image
    rectFilter = tf.ones(shape=[*resize_ratio, 1, 1], dtype=image.dtype)
    outimage = tf.nn.conv2d(image, rectFilter, strides=[1, *resize_ratio, 1], padding="VALID")

    return outimage


def resample_intensity_sensor(intensity, resize_ratio):
    """Helper function calling the desired resize_area method to process the passed in intensity.

    Currently, only an autodifferentiable, non-fractional box-filter area-resize method is encoded and used.
    In the future, this function may be expanded to take other options.

    Args:
        `Intensity` (tf.float64): Field intensity to be resampled and box integrated, of shape (batch_size, Ny, Nx, channel_size)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)
    Returns:
        `tf.float`: Area-resized intensity of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    return resize_area(intensity, resize_ratio)


def resample_phase_sensor(phase, resize_ratio):
    """Helper function calling the desired resize_area average method to process the passed in phase profile.

    Args:
        `Phase` (tf.float64): Field phase profile to be resampled and box averaged, of shape (batch_size, Ny, Nx, channel_size)
        `Resize_ratio` (tuple): Resize factor along y and x, defined by (scaley, scalex)
    Returns:
        `tf.float`: Area-resized phase of shape (batch_size, Ny/resize_ratio["y"], Nx/resize_ratio["x"], channel_size).
    """
    # we want the output to be the average phase over the larger pixel rather than the sum
    phasex = resize_area(tf.math.cos(phase), resize_ratio) / resize_ratio[1] / resize_ratio[0]
    phasey = resize_area(tf.math.sin(phase), resize_ratio) / resize_ratio[1] / resize_ratio[0]

    return tf.math.atan2(phasey, phasex)


def sensorMeasurement_intensity_phase(sensor_intensity, sensor_phase, parameters, use_radial):
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
    input_rank = len(sensor_intensity.shape)
    init_shape = sensor_intensity.shape
    if input_rank == 1:
        raise ValueError("Input tensors must have a rank \geq 2")
    elif input_rank == 2:
        sensor_intensity = tf.expand_dims(sensor_intensity, 0)
        sensor_phase = tf.expand_dims(sensor_phase, 0)
    elif input_rank > 3:
        sensor_intensity = tf.reshape(sensor_intensity, [-1, init_shape[-2], init_shape[-1]])
        sensor_phase = tf.reshape(sensor_phase, [-1, init_shape[-2], init_shape[-1]])

    ## Unpack parameters
    # If we are doing PSFs, then we would have passed use_radial = False because we always want 2D PSFs
    # IF we are doing field propagation, we might instead want to keep vectors as radial
    calc_sensor_dx_m = parameters["calc_sensor_dx_m"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    sensor_pixel_number = parameters["sensor_pixel_number"]
    radial_flag = use_radial
    dtype = parameters["dtype"]

    ### Only do convolutional downdampling if the grid size and the sensor sizes are not the same (up to a tolerance)
    tol = 0.2e-6
    diff_x = abs(sensor_pixel_size_m["x"] - calc_sensor_dx_m["x"])
    diff_y = abs(sensor_pixel_size_m["y"] - calc_sensor_dx_m["y"])
    if diff_x > tol or diff_y > tol:
        # Because we will do convolutional downsampling, we first want to take the field and reinterpolate it onto a new grid
        # so that the downsample mount is close to an integer instead of a float
        scalex = sensor_pixel_size_m["x"] / calc_sensor_dx_m["x"]
        scaley = sensor_pixel_size_m["y"] / calc_sensor_dx_m["y"]
        upsample_size = (
            int(np.round(scaley) / scaley * init_shape[-2]),
            int(np.round(scalex) / scalex * init_shape[-1]),
        )
        resize_ratio = (int(np.round(scaley)), int(np.round(scalex)))

        # resize needs 4-D tensor of shape [batch, height, width, channels]
        sensor_phase = tf.expand_dims(sensor_phase, -1)
        sensor_intensity = tf.expand_dims(sensor_intensity, -1)

        sensor_phase_real = tf.image.resize(tf.math.cos(sensor_phase), upsample_size, method="bicubic")
        sensor_phase_imag = tf.image.resize(tf.math.sin(sensor_phase), upsample_size, method="bicubic")
        sensor_intensity = tf.image.resize(sensor_intensity, upsample_size, method="bicubic")
        sensor_phase = tf.math.atan2(sensor_phase_imag, sensor_phase_real)

        sensor_intensity = tf.squeeze(resample_intensity_sensor(sensor_intensity, resize_ratio), -1)
        sensor_phase = tf.squeeze(resample_phase_sensor(sensor_phase, resize_ratio), -1)

    # Crop or pad with zeros to the sensor size
    if radial_flag:
        output_size = {"r": sensor_pixel_number["r"], "y": 1}
    else:
        output_size = {"x": sensor_pixel_number["x"], "y": sensor_pixel_number["y"]}
    sensor_intensity = radial_conditional_resize_with_crop_or_pad(sensor_intensity, radial_flag, output_size)
    sensor_phase = radial_conditional_resize_with_crop_or_pad(sensor_phase, radial_flag, output_size)

    # Return with the same batch_size shape
    new_shape = sensor_intensity.shape
    if input_rank != 3:
        sensor_intensity = tf.reshape(sensor_intensity, [*init_shape[:-2], *new_shape[-2:]])
        sensor_phase = tf.reshape(sensor_phase, [*init_shape[:-2], *new_shape[-2:]])

    return sensor_intensity, sensor_phase


# def batch_interp_regular_nd(real_refdat_2d, ref_grid_min, ref_grid_max, x_grid_mesh, y_grid_mesh):
#     """Helper function when calling batch_interp_regular_nd_grid.

#     Args:
#         real_refdat_2d (tf.float): Real-valued tensor corresponding to reference ydata on ref_grid_min, ref_grid_max of
#             shape (batch_size, Ny, Nx).
#         ref_grid_min (list): n=2 element list specifying the min of the reference grid along each dimension.
#         ref_grid_max (_type_): n=2 element list specifying the max of the reference grid along each dimension
#         x_grid_mesh (_type_): Rank n=2 tensor corresponding to the x coordinate of each pixel
#         y_grid_mesh (_type_): Rank n=2 tensor corresponding to the y coordinate of each pixel
#         parameters (_type_): _description_
#     """

#     # x is a list of coordinates to interpolate on of shape [..., D, nd] where nd is the space dimensionality and D
#     # is the total number of points. This is to match the required input of the tensorflow call
#     x = tf.stack([tf.reshape(y_grid_mesh, -1), tf.reshape(x_grid_mesh, -1)], axis=1)
#     interp_dat = tfp.math.batch_interp_regular_nd_grid(x, ref_grid_min, ref_grid_max, real_refdat_2d, axis=-2)

#     return tf.reshape(interp_dat, shape=(real_refdat_2d.shape[0], x_grid_mesh.shape[0], x_grid_mesh.shape[1]))
