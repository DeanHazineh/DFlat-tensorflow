import tensorflow as tf


def latent_to_param(z_latent, pmin=0, pmax=1):
    """Takes a latent tensor (defined on domain R[-inf, inf]) and returns the corresponding param tensor
        (defined on domain R[pmin, pmax]).

    Args:
        `z_latent` (tf.float): Latent tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: param tensor of equivalent shape

    """
    return ((tf.math.tanh(tf.clip_by_value(z_latent, -10, 10)) + 1) / 2 * (pmax - pmin)) + pmin


def param_to_latent(p_param, pmin=0, pmax=1):
    """Takes a param tensor (defined on domain R[pmin, pmax]) and returns the corresponding latent tensor
    (defined on domain R[-inf, inf]).

    Args:
        `p_param` (tf.float): Param tensor
        `pmin` (tf.float, optional): minimum param value. Defaults to 0.
        `pmax` (tf.float, optional): maximum param value. Defaults to 1.

    Returns:
        `tf.float`: Latent tensor of equivalent shape
    """
    return tf.math.atanh((tf.clip_by_value(p_param, pmin, pmax) - pmin) / (pmax - pmin) * 2 - 1)
