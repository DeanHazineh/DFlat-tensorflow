import tensorflow as tf
import numpy as np
from dflat.render_layer import fourier_convolve
from .latent_param_utils import *


def tf_rotate_coord(xcart, ycart, theta):
    xr = xcart * tf.math.cos(theta) + ycart * tf.math.sin(theta)
    yr = -xcart * tf.math.sin(theta) + ycart * tf.math.cos(theta)
    return xr, yr


def tf_assemble_bounded_rectangle(P, hx, hy, fm, return_bbox=False):
    ## Determine range of valid angles and select angle
    hfm = fm / 2
    dmin = tf.math.minimum(hx, hy)
    diag = np.sqrt(2) * hfm
    dtheta = np.pi / 2 - tf.math.acos(tf.clip_by_value(dmin / diag, 0, 1))
    theta = dtheta * (2 * P[2] - 1)

    ## Determine height range and selection
    hmax = tf.math.maximum(tf.math.minimum((hy - hfm * tf.math.abs(tf.math.sin(theta))) / tf.math.cos(theta), (hx - hfm * tf.math.cos(theta)) / tf.math.abs(tf.math.sin(theta))), hfm)
    hh = hfm + P[3] * (hmax - hfm)

    ### Determine width range and selection
    ### This is the same rule as above but with x and y dimensions swapped
    wmax = tf.math.minimum((hx - hh * tf.math.abs(tf.math.sin(theta))) / tf.math.cos(theta), (hy - hh * tf.math.cos(theta)) / tf.math.abs(tf.math.sin(theta)))
    hw = hfm + P[4] * (wmax - hfm)

    if return_bbox:
        # Get bounding box
        diag = tf.math.sqrt(hh**2 + hw**2)
        psi = tf.math.atan(hh / hw)
        bbox_hw = diag * tf.math.cos(psi - tf.math.abs(theta))
        bbox_hh = diag * tf.math.sin(np.pi - tf.math.abs(psi) - tf.math.abs(theta))

        return hh, hw, theta, bbox_hw, bbox_hh
    else:
        return hh, hw, theta


def tf_rectangle_binary(Ux, Uy, Nx, Ny, cx, cy, phi, theta, hw, hh, exp_power, sigmoid_coefficient):
    xc, yc = tf.meshgrid(tf.linspace(-Ux / 2, Ux / 2, Nx), tf.linspace(-Uy / 2, Uy / 2, Ny))
    xc = tf.cast(xc, cx.dtype)
    yc = tf.cast(yc, cy.dtype)

    cxr, cyr = tf_rotate_coord(cx - Ux / 2, cy - Uy / 2, phi)
    x1, y1 = tf_rotate_coord(xc - cxr, yc + cyr, theta + phi)
    val = 1 - tf.math.abs(x1 / hw) ** exp_power - tf.math.abs(y1 / hh) ** exp_power
    val = tf.math.sigmoid(sigmoid_coefficient * val)

    return val


def morphological_filter(binary, kernel, mode="closing"):
    if tf.rank(kernel) == 2:
        kernel = kernel[:, :, tf.newaxis]
    if tf.rank(binary) == 2:
        x = binary[tf.newaxis, :, :, tf.newaxis]
    elif tf.rank(binary == 4):
        x = binary

    if mode == "closing":
        x = tf.nn.dilation2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1])
        x = tf.nn.erosion2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1])
    elif mode == "opening":
        x = tf.nn.erosion2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1])
        x = tf.nn.dilation2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC", dilations=[1, 1, 1, 1])
    else:
        raise ValueError("Mode must be one of 'closing' or 'opening'")

    return x[0, :, :, 0]


def check_binary(binary):
    # return tf.where(tf.math.is_nan(binary), tf.zeros_like(binary),  binary)
    return binary


def tf_generate_parametric_freeform4(P1, P2, P3, P4, constraint_dict):
    b = constraint_dict["brule"]
    fm = constraint_dict["fm"]
    Ux = constraint_dict["Ux"]
    Uy = constraint_dict["Uy"]
    Nx = constraint_dict["Nx"]
    Ny = constraint_dict["Ny"]
    mode = constraint_dict["constraint_type"]
    sigmoid_coefficient = 100
    exp_power = 20
    kw = fm * 0.90

    Plist = [P1, P2, P3, P4]
    binary = []
    for P in Plist:
        # Get center coordinate
        cx1 = (b + fm / 2) + P[0] * (Ux - 2.0 * b - fm)
        cy1 = (b + fm / 2) + P[1] * (Uy - 2.0 * b - fm)

        # Identify the max bounding box
        hx1 = tf.math.minimum(cx1 - b, Ux - b - cx1)
        hy1 = tf.math.minimum(Uy - b - cy1, cy1 - b)

        # Get fin within the bounding box and make the binary mask
        hh1, hw1, theta1 = tf_assemble_bounded_rectangle(P, hx1, hy1, fm, False)
        binary.append(
            tf.expand_dims(
                tf_rectangle_binary(Ux, Uy, Nx, Ny, cx1, cy1, tf.constant(0.0, dtype=theta1.dtype), theta1, hw1, hh1, exp_power, sigmoid_coefficient),
                0,
            )
        )

    ### Form the joint binary and convolve with function
    binary = tf.concat(binary, axis=0)
    binary_sum = tf.math.sigmoid((tf.reduce_sum(binary, axis=0) - 0.5) * sigmoid_coefficient)
    if mode == "raw_binary":
        return check_binary(binary_sum)

    xx, yy = np.meshgrid(np.linspace(-Ux, Ux, Nx), np.linspace(-Uy, Uy, Ny))
    kernel = 1 - tf.math.abs(xx * 2 / kw) ** 2 - tf.math.abs(yy * 2 / kw) ** 2
    kernel = tf.math.sigmoid(sigmoid_coefficient * kernel)
    kernel = tf.cast(kernel, binary_sum.dtype)

    if mode == "convolutional":
        kernel = kernel / tf.math.reduce_sum(kernel)
        binary_sum = tf.math.abs(fourier_convolve(binary_sum, kernel))
        binary_sum = tf.math.sigmoid((binary_sum - 0.5) * sigmoid_coefficient)
    elif mode == "morphological":
        binary_sum = morphological_filter(binary_sum, kernel, mode="closing")
        # cell = morphological_filter(cell, kernel, mode="opening")
    else:
        raise ValueError("mode must be either morphological or convolutional")

    return check_binary(binary_sum)


def tf_generate_parametric_freeform(Plist, constraint_dict):
    b = constraint_dict["brule"]
    fm = constraint_dict["fm"]
    Ux = constraint_dict["Ux"]
    Uy = constraint_dict["Uy"]
    Nx = constraint_dict["Nx"]
    Ny = constraint_dict["Ny"]
    mode = constraint_dict["constraint_type"]
    sigmoid_coefficient = 100
    exp_power = 20
    kw = fm * 0.90

    # Plist = [latent_to_param(z) for z in latent_list]
    binary = []
    for P in Plist:
        # Get center coordinate
        cx1 = (b + fm / 2) + P[0] * (Ux - 2.0 * b - fm)
        cy1 = (b + fm / 2) + P[1] * (Uy - 2.0 * b - fm)

        # Identify the max bounding box
        hx1 = tf.math.minimum(cx1 - b, Ux - b - cx1)
        hy1 = tf.math.minimum(Uy - b - cy1, cy1 - b)

        # Get fin within the bounding box and make the binary mask
        hh1, hw1, theta1 = tf_assemble_bounded_rectangle(P, hx1, hy1, fm, False)
        binary.append(
            tf.expand_dims(
                tf_rectangle_binary(Ux, Uy, Nx, Ny, cx1, cy1, tf.constant(0.0, dtype=theta1.dtype), theta1, hw1, hh1, exp_power, sigmoid_coefficient),
                0,
            )
        )

    ### Form the joint binary and convolve with function
    binary = tf.concat(binary, axis=0)
    binary_sum = tf.math.sigmoid((tf.reduce_sum(binary, axis=0) - 0.5) * sigmoid_coefficient)
    if mode == "raw_binary":
        return check_binary(binary_sum)

    xx, yy = np.meshgrid(np.linspace(-Ux, Ux, Nx), np.linspace(-Uy, Uy, Ny))
    kernel = 1 - tf.math.abs(xx * 2 / kw) ** 2 - tf.math.abs(yy * 2 / kw) ** 2
    kernel = tf.math.sigmoid(sigmoid_coefficient * kernel)
    kernel = tf.cast(kernel, binary_sum.dtype)
    if mode == "convolutional":
        kernel = kernel / tf.math.reduce_sum(kernel)
        binary_sum = tf.math.abs(fourier_convolve(binary_sum, kernel))
        binary_sum = tf.math.sigmoid((binary_sum - 0.5) * sigmoid_coefficient)
    elif mode == "morphological":
        binary_sum = morphological_filter(binary_sum, kernel, mode="closing")
        # binary_sum = morphological_filter(binary_sum, kernel, mode="opening")
    else:
        raise ValueError("mode must be either morphological or convolutional")

    return check_binary(binary_sum)
