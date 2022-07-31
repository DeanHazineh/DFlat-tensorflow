import dflat.image_layer.core.freiburg_imroutine as imroutine
import dflat.image_layer.core.image_gen_utilities as im_utils
import dflat.image_layer.core.spectral_utilities as spec_utils
import random
import numpy as np
import tensorflow as tf
import cv2 as cv

rgb_weights = [0.2989, 0.5870, 0.1140]


# It is the job of dataGenerators to return meaningful training data and with a size that is consistent to the PSF
# sizes in the rendering layer! This simplifies general code pipelines and makes it easier to reuse or modify pipelines.
def RGB_to_spectral(rgb_image, wavelength_nm):
    """Converts RGB image of shape [RGB,Ny,Nx] to spectral form of [N_lambda, Ny, Nx].
    Args:
        rgb_image (float): RGB image where first dimension is the RGB channel vector.
        wavelength_nm (list): List of wavelength values to gen spectral content at
    """

    if rgb_image.shape[0] != 3:
        raise ValueError("rgb_image must have channel len 3 in first dimension.")

    rgb_bar = spec_utils.get_rgb_bar_CIE1931(wavelength_nm)
    rgb_bar = np.transpose(rgb_bar, [1, 0])
    rgb_image = np.transpose(rgb_image, [1, 2, 3, 4, 0])

    return np.transpose(np.matmul(rgb_image, rgb_bar), [4, 0, 1, 2, 3])


def cv_resize_crop_or_pad(im, resize_dim, crop_dim):
    # im has shape (H, W, Channels)
    im = cv.resize(im, resize_dim, interpolation=cv.INTER_AREA)
    if len(im.shape) == 2:
        im = tf.expand_dims(im, -1)

    im = tf.squeeze(tf.image.resize_with_crop_or_pad(tf.expand_dims(im, 0), crop_dim["y"], crop_dim["x"]), 0)

    return tf.squeeze(im)


def dataGenerator_FT_depth(
    max_num_layers,
    min_map_val,
    max_map_val,
    sensor_dim,
    mode="test",
    set="",
    grayscale=True,
    dtype=tf.float64,
    batchSize=5,
):
    # specify train vs test tag
    if not (mode == "test" or mode == "train"):
        raise ValueError("mode must be test or train")

    # specify set
    if not set in ["A", "B", "C"]:
        raise ValueError("set must be either 'A', 'B', 'C'")

    # Configure generator to the particular dataset
    if mode == "test":
        tag = "TEST"
        maxFold = 149
    else:
        tag = "TRAIN"
        maxFold = 749
    item_range = [6, 15]

    # Point to flying things data path
    ft_impath = (
        "dflat/datasets_image/flyingthings3d__frames_cleanpass_webp/frames_cleanpass_webp/" + tag + "/" + set + "/"
    )
    ft_objpath = "dflat/datasets_image/flyingthings3d__object_index/object_index/" + tag + "/" + set + "/"
    ft_matpath = "dflat/datasets_image/flyingthings3d__material_index/material_index/" + tag + "/" + set + "/"

    call_counter = 0
    batch_counter = 0
    while True:
        im_stack = []
        im_mask_stack = []
        im_map_stack = []
        im_mat_stack = []
        depthVal_seed = np.random.randint(0, 1e9)
        while batch_counter < batchSize:

            if call_counter > maxFold:
                call_counter = 0

            try:
                chooseFold = f"{call_counter:04d}"
                ster = "left" if random.choice([True, False]) else "right"
                chooseIm = f"{random.randint(item_range[0], item_range[1]):04d}"
                impath = chooseFold + "/" + ster + "/" + chooseIm

                im = np.asarray(imroutine.read(ft_impath + impath + ".webp")) / 256
                if grayscale:
                    im = np.expand_dims(np.dot(im, rgb_weights), -1)

                im_obj = np.asarray(imroutine.read(ft_objpath + impath + ".pfm"))
                im_mat = np.asarray(imroutine.read(ft_matpath + impath + ".pfm"))
                batch_counter += 1
            except:
                call_counter += 1
                continue

            # First resize to take up the sensor dimension while preserving aspect ratio
            ratiox = int(np.ceil(sensor_dim["x"] / im.shape[1]))
            ratioy = int(np.ceil(sensor_dim["y"] / im.shape[0]))
            resize_ratio = ratiox if ratiox > ratioy else ratioy
            dim = (im.shape[1] * resize_ratio, im.shape[0] * resize_ratio)

            im = cv_resize_crop_or_pad(im, dim, sensor_dim).numpy()
            if grayscale:
                im = np.expand_dims(im, -1)
            im_obj = cv_resize_crop_or_pad(np.expand_dims(im_obj, -1), dim, sensor_dim).numpy()
            im_mat = cv_resize_crop_or_pad(np.expand_dims(im_mat, -1), dim, sensor_dim).numpy()

            im_mask = im_utils.im_obj_to_masks(im_obj, max_num_layers)
            depth_layers, im_map = im_utils.gen_depth_map(
                im_obj, max_num_layers, min_map_val, max_map_val, depthVal_seed
            )
            call_counter += 1

            ### Restructure scene data to match PSF dimensions
            im = np.transpose(im, [2, 0, 1])
            im = tf.cast(im[:, tf.newaxis, tf.newaxis, :, :], dtype)
            im_mask = tf.cast(im_mask[tf.newaxis, tf.newaxis, :, :, :], dtype)
            im_map = tf.expand_dims(im_map, 0)
            im_mat = tf.expand_dims(im_mat, 0)

            im_stack.append(im)
            im_mask_stack.append(im_mask)
            im_map_stack.append(im_map)
            im_mat_stack.append(im_mat)

        # print(im_stack)
        batch_counter = 0
        im_stack = np.stack(im_stack)
        im_mask_stack = np.stack(im_mask_stack)
        im_map_stack = np.stack(im_map_stack)
        im_mat_stack = np.stack(im_mat_stack)

        yield im_stack, im_mask_stack, im_map_stack, im_mat_stack, depth_layers
