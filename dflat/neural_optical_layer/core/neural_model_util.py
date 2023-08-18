import tensorflow as tf
import numpy as np

from .arch_DNN import *
from .caller_MLP import *

listModelNames = mlp_model_names


def list_models():
    print(listModelNames)
    return


def load_neuralModel(model_selection_string, dtype=tf.float32):
    if model_selection_string not in listModelNames:
        raise ValueError("mlp_layer: requested MLP is not one of the supported libraries")
    else:
        mlp = globals()[model_selection_string]
        mlp = mlp(dtype)
        mlp.customLoadCheckpoint()
        mlp.trainable = False

    print("Loaded Model: ", mlp.get_model_name())
    return mlp


###
def convert_shape_to_param(shape_vector, MLP_model):
    """(Helper function for neural optical models) Given an unormalized shape vector and an MLP model,
    return the normalized params, in [0,1].

    Args:
        `shape_vector` (np.float): Unnormalized shape vector for a cell, of form (N, D) where D is the shape degree.
        `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

    Returns:
        `np.float`: Model normalized parameter vector suitable for mlp input
    """
    paramList = [shape_vector[:, i : i + 1] for i in range(shape_vector.shape[1])]

    return tf.transpose(tf.stack(MLP_model.normalizeInput(paramList)))


def convert_param_to_shape(norm_param, MLP_model):
    """(Helper function for neural optical models) Given a model normalized parameter array, return the unnormalized
    shape vector which corresponds to structure lengths in m.

    Args:
        `norm_param` (np.float): Normalized parameters for a cell, of form (N, D) where D is the shape degree.
        `MLP_model` (MLP_Object): A pre-trained neural optical model in DFlat.

    Returns:
        `np.float`: The unnormalized parameter vector, where lengths are back in meaningful units of m.
    """
    databounds = MLP_model.get_preprocessDataBounds()
    shapeDegree = len(databounds) - 1
    # return tf.transpose(tf.stack([norm_param[:, i] * (databounds[i][1] - databounds[i][0]) + databounds[i][0] for i in range(shapeDegree)]))
    return np.array([norm_param[:, i] * (databounds[i][1] - databounds[i][0]) + databounds[i][0] for i in range(shapeDegree)]).T


def init_norm_param(init_type, dtype, gridShape, mlp_input_shape, init_args=[]):
    if init_type == "uniform":
        norm_param = 0.5 * tf.ones(shape=(mlp_input_shape - 1, gridShape[1], gridShape[2]), dtype=dtype)

    elif init_type == "random":
        norm_param = tf.random.uniform(shape=(mlp_input_shape - 1, gridShape[1], gridShape[2]), dtype=dtype)

    else:
        raise ValueError("initialize_norm_param: invalid init_type string;")

    return norm_param


def flatten_reshape_shape_parameters(shape_vector):
    """Takes a shape/param vector of (D, PixelsY, PixelsX) and flattens to shape (PixelsY*PixelsX,D)"""
    vec_shape = tf.shape(shape_vector)
    return tf.reshape(tf.transpose(shape_vector, [1, 2, 0]), [vec_shape[1] * vec_shape[2], -1])
