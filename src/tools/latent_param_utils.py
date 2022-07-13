import tensorflow as tf


def latent_to_param(z_latent):
    """Takes a latent tensor (defined on domain R[-inf, inf]) and returns the corresponding param tensor 
        (defined on domain R[0,1]).

    Args:
        `z_latent` (tf.float): Latent tensor

    Returns:
        `tf.float`: param tensor of equivalent shape
    """
    return (tf.math.tanh(z_latent) + 1) / 2


def param_to_latent(p_param):
    """Takes a param tensor (defined on domain R[0,1]) and returns the corresponding latent tensor
    (defined on domain R[-inf, inf]).

    Args:
        `p_param` (tf.float): Param tensor

    Returns:
        `tf.float`: Latent tensor of equivalent shape
    """
    return tf.math.atanh(tf.clip_by_value(p_param, 0.0, 1.0) * 2 - 1)
