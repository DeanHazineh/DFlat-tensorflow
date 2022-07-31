import numpy as np
import math


def im_obj_to_masks(im_obj, max_num_layers):
    """_summary_

    Args:
        im_obj (float): Image array of shape (Ny, Nx) where each value is a semantic idenifier for the pixel.
        max_num_layers (int): Maximum number of semantic layers to produce, by grouping identifiers into sets

    Returns:
        float:
    """

    # im_obj is an image where each object in the scene is assigned a unique identifier (float or int indicator)
    im_obj = im_obj.astype(int)
    seg_mask_list = []
    layer_idx = np.unique(im_obj)
    idx_step = math.floor(len(layer_idx) / max_num_layers)
    for iter in range(max_num_layers):
        if iter == (max_num_layers - 1):
            idx_set = layer_idx[iter * idx_step :]
        else:
            idx_set = layer_idx[iter * idx_step : (iter + 1) * idx_step]
        seg_mask = np.where(np.isin(im_obj, idx_set, assume_unique=True), 1, 0)
        seg_mask_list.append(np.expand_dims(seg_mask, 0))

    return np.vstack(seg_mask_list)


def gen_depth_map(im_obj, max_num_layers, min_depth, max_depth, seed):
    # Get seg_mask from im_obj
    seg_mask = im_obj_to_masks(im_obj, max_num_layers)
    seg_mask = np.array(seg_mask)

    # ## Cast binary seg_mask to depth map
    # num_layers = len(seg_mask)
    # np.random.seed(seed)
    # depth = np.sort(np.random.uniform(min_depth, max_depth, num_layers))[::-1]

    ## try linearly spaced map
    num_layers = len(seg_mask)
    depth = np.linspace(min_depth, max_depth, num_layers)[::-1]

    return depth, np.sum(seg_mask * np.expand_dims(np.expand_dims(depth, -1), -1), axis=0)
