import tensorflow as tf
from image_layer.core.fronto_planar_render import *


class Fronto_Planar_Renderer_Sparse(tf.keras.layers.Layer):
    """_summary_

    Args:
        tf (_type_): _description_
    """

    def __init__(self,):
        """_summary_
        """
        super(Fronto_Planar_Renderer_Sparse, self).__init__()

    def __call__(self, psf_intensity, focus_image, seg_mask):
        """ Renders the detector image given, a PSF, segmentation_maks, and corresponding PSFs. Currently, this function
        only incorporates the most basic, naive, depth rendering image formation model.

        Args:
            'psf_intensity' (tf.float): PSF for rendering, of shape (batch or None, N_lambda, N_b, N_z, Ny, Nx).
            'focus_image' (tf.float): Pinhole image of shape (batch or None, , N_lambda, N_b, 1, Ny, Nx), dimension broadcast allowed.
            'seg_mask' (tf.float): Segmentation masks for scene, of shape (batch or None, , 1, 1, N_z, Ny, Nx)

        Returns:
            tf.float: Rendered sensor image of shape, (N_b, Ny, Nx)
        """

        # Ensure tensor inputs
        dtype = psf_intensity.dtype
        if not tf.is_tensor(focus_image):
            focus_image = tf.convert_to_tensor(focus_image, dtype=dtype)

        if not tf.is_tensor(psf_intensity):
            psf_intensity = tf.convert_to_tensor(psf_intensity, dtype=dtype)

        if not tf.is_tensor(seg_mask):
            seg_mask = tf.convert_to_tensor(seg_mask, dtype=dtype)

        return render_depth_naive(psf_intensity, focus_image, seg_mask)

