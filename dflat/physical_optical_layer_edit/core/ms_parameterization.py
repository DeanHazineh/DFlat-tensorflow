import numpy as np
import tensorflow as tf


def get_cartesian_grid(Lx, Nx, Ly, Ny):
    return np.meshgrid(np.linspace(-Lx / 2, Lx / 2, Nx), np.linspace(-Ly / 2, Ly / 2, Ny))


def build_rectangle_resonator(norm_param, feature_layer, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay):
    # Single layer metasurface
    # norm_param: A 'tf.Tensor' of shape (pixelsX, pixelsY, 2)
    # Binary like (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)

    # unpack inputs
    dtype = norm_param.dtype
    TF_ZERO = tf.constant(0.0, dtype=dtype)
    norm_px = norm_param[:, :, 0]
    norm_py = norm_param[:, :, 1]
    span_max = tf.cast(span_limits["max"], dtype=dtype)
    span_min = tf.cast(span_limits["min"], dtype=dtype)

    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = r_x[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    r_y = r_y[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]

    ## Generate Rectangle fin shape
    r1 = 1 - tf.math.abs(x_mesh * 2.0 / r_x) ** 15.0 - tf.math.abs(y_mesh * 2.0 / r_y) ** 15.0
    r1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * r1), TF_ZERO)

    struct_binaries = [None for i in range(Nlay)]
    struct_binaries[feature_layer] = r1
    return struct_binaries


def build_coupled_rectangular_resonators(norm_param, feature_layer, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay):
    # norm_param: A 'tf.Tensor' of shape (pixelsX, pixelsY, 4, 2)
    POWER_EXP = 40
    TF_ZERO = tf.constant(0.0, dtype=tf.float32)

    # unpack inputs
    norm_px = tf.transpose(norm_param[:, :, :, 0], [2, 0, 1])
    norm_py = tf.transpose(norm_param[:, :, :, 1], [2, 0, 1])
    span_max = tf.cast(span_limits["max"], dtype=tf.float32)
    span_min = tf.cast(span_limits["min"], dtype=tf.float32)

    # Nanopost width ranges
    r_x = (norm_px * (span_max - span_min) + span_min) * Lx
    r_y = (norm_py * (span_max - span_min) + span_min) * Ly
    r_x = r_x[:, tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    r_y = r_y[:, tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]

    c1 = 1 - tf.math.abs((x_mesh + Lx / 4) * 2 / r_x[0]) ** POWER_EXP - tf.math.abs((y_mesh + Ly / 4) * 2 / r_y[0]) ** POWER_EXP
    c2 = 1 - tf.math.abs((x_mesh + Lx / 4) * 2 / r_x[1]) ** POWER_EXP - tf.math.abs((y_mesh - Ly / 4) * 2 / r_y[1]) ** POWER_EXP
    c3 = 1 - tf.math.abs((x_mesh - Lx / 4) * 2 / r_x[2]) ** POWER_EXP - tf.math.abs((y_mesh + Ly / 4) * 2 / r_y[2]) ** POWER_EXP
    c4 = 1 - tf.math.abs((x_mesh - Lx / 4) * 2 / r_x[3]) ** POWER_EXP - tf.math.abs((y_mesh - Ly / 4) * 2 / r_y[3]) ** POWER_EXP
    c1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c1), TF_ZERO)
    c2 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c2), TF_ZERO)
    c3 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c3), TF_ZERO)
    c4 = tf.complex(tf.math.sigmoid(sigmoid_coeff * c4), TF_ZERO)

    struct_binaries = [None for i in range(Nlay)]
    struct_binaries[feature_layer] = c1 + c2 + c3 + c4

    return struct_binaries


def build_cylindrical_nanoposts(norm_param, feature_layer, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay):
    dtype = norm_param.dtype
    TF_ZERO = tf.constant(0.0, dtype=dtype)

    # unpack inputs
    norm_pr = norm_param[:, :, 0]
    span_max = tf.cast(span_limits["max"], dtype=dtype)
    span_min = tf.cast(span_limits["min"], dtype=dtype)

    print(tf.math.minimum(Lx, Ly))
    print((norm_pr * (span_max - span_min) + span_min))
    radius = (norm_pr * (span_max - span_min) + span_min) * tf.math.minimum(Lx, Ly)
    radius = radius[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis]

    r1 = 1 - (x_mesh / radius) ** 2 - (y_mesh / radius) ** 2
    r1 = tf.complex(tf.math.sigmoid(sigmoid_coeff * r1), TF_ZERO)

    struct_binaries = [None for i in range(Nlay)]
    struct_binaries[feature_layer] = r1
    return struct_binaries


def generate_cell_perm(norm_param, rcwa_parameters, parameterization_type, feature_layer):
    """
    Generates permittivity and permeability for a unit cell comprising of structures according to "parameterization_type"
    set in the rcwa_parameters setting dict.

    Args:
        `norm_param` (tf.float): A tensor of shape (pixelsY, pixelsX, d), where d are normalized shape parameters
        `rcwa_parameters`: A dict of type `rcwa_params` containing simulation and optimization settings.

    Returns:
        `tf.float`: A tensor of shape (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny) specifying the relative permittivity
            distribution of each cell.
        `tf.float`: A tensor of shape (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny) specifying the relative permeability
            distribution of each cell.
    """

    # Retrieve simulation size parameters
    batchSize = rcwa_parameters["batchSize"]
    pixelsX = rcwa_parameters["pixelsX"]
    pixelsY = rcwa_parameters["pixelsY"]
    Nlay = rcwa_parameters["Nlay"]
    Nx = rcwa_parameters["Nx"]
    Ny = rcwa_parameters["Ny"]
    Lx = rcwa_parameters["Lx"]
    Ly = rcwa_parameters["Ly"]
    dtype = rcwa_parameters["dtype"]
    cdtype = rcwa_parameters["cdtype"]
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    materials_shape_lay = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)

    # Define the cartesian cross section.
    # Convert to tensors and expand and tile to match the simulation shape.
    # Permittivity and permeability like (batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)
    x_mesh, y_mesh = get_cartesian_grid(Lx, Nx, Ly, Ny)
    y_mesh = tf.cast(y_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :], dtype)
    y_mesh = tf.tile(y_mesh, multiples=(batchSize, pixelsX, pixelsY, 1, 1, 1))
    x_mesh = tf.cast(x_mesh[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :], dtype)
    x_mesh = tf.tile(x_mesh, multiples=(batchSize, pixelsX, pixelsY, 1, 1, 1))

    # Initialize relative permeability.
    UR = rcwa_parameters["urd"] * tf.ones(materials_shape, dtype=cdtype)

    # Initialize the relative permittivity
    init_function = ALLOWED_PARAMETERIZATION_TYPE[parameterization_type]
    span_limits = DEFAULT_SPAN_LIMITS[parameterization_type]
    lay_eps_list = rcwa_parameters["lay_eps_list"]
    sigmoid_coeff = 1000.0
    struct_binary = init_function(norm_param, feature_layer, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay)

    ER = []
    for i in range(Nlay):
        if struct_binary[i] != None:
            ER.append(lay_eps_list[i] + (rcwa_parameters["erd"] - lay_eps_list[i]) * struct_binary[i])
        else:
            ER.append(lay_eps_list[i] * tf.ones(materials_shape_lay, dtype=cdtype))
    ER = tf.concat(ER, axis=3)

    return ER, UR


def list_cell_parameterizations():
    print(list(ALLOWED_PARAMETERIZATION_TYPE.keys()))
    return


ALLOWED_PARAMETERIZATION_TYPE = {
    "rectangular_resonator": build_rectangle_resonator,
    "coupled_rectangular_resonator": build_coupled_rectangular_resonators,
    "cylindrical_nanopost": build_cylindrical_nanoposts,
}

CELL_SHAPE_DEGREE = {
    "rectangular_resonator": [2],
    "coupled_rectangular_resonator": [4, 2],
    "cylindrical_nanopost": [1],
}

DEFAULT_SPAN_LIMITS = {
    "rectangular_resonator": {"min": 0.05, "max": 0.95},
    "coupled_rectangular_resonator": {"min": 0.05, "max": 0.20},
    "cylindrical_nanopost": {"min": 0.05, "max": 0.95},
}
