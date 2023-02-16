import numpy as np
import cv2 as cv


def load_png_as_grayscale(file_path, sensor_dim=None, resize_method="crop"):
    ### read in image as grayscale with CV and make (H, W, Channels=1)
    img = np.asarray(cv.imread(str(file_path), cv.IMREAD_GRAYSCALE))
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    ### Resize while preserving aspect ratio if sensor_dimensions is passed in
    if sensor_dim is not None:
        img = cv_aspectResize(img, sensor_dim, resize_method)

    return img


def cv_aspectResize(img, sensor_dim, resize_method):
    ### Select resize dimension
    im_shape = img.shape
    ratiox = sensor_dim["x"] / im_shape[1]
    ratioy = sensor_dim["y"] / im_shape[0]

    if resize_method == "crop":
        # resize larger than the sensor then center crop to the sensor dimension
        resize_ratio = np.maximum(ratiox, ratioy)
    elif resize_method == "pad":
        resize_ratio = np.minimum(ratiox, ratioy)
    else:
        raise ValueError("resize method must be either 'crop' or 'pad'")

    resize_dim = (int(im_shape[1] * resize_ratio), int(im_shape[0] * resize_ratio))
    img = cv.resize(img, resize_dim, interpolation=cv.INTER_CUBIC)

    ### resize with crop or pad to sensor dimension now
    img = center_crop_or_pad(img, sensor_dim)

    return img


def center_crop_or_pad(img, new_dim):
    targ_width, targ_height = new_dim["x"], new_dim["y"]

    ### For safety, if channel dimension is not included then add
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)

    ### Run cropping if requested demension is smaller
    height, width = img.shape[0], img.shape[1]
    cx, cy = 0, 0
    if targ_width < width:  # crop
        cx = int((width - targ_width) / 2)
    if targ_height < height:
        cy = int((height - targ_height) / 2)
    img = img[cy : height - cy, cx : width - cx, :]

    ### Pad if height and width are smaller than the target
    height, width = img.shape[0], img.shape[1]
    dx = targ_width - width
    dy = targ_height - height

    padx = (0, 0)
    pady = (0, 0)
    if dx > 0:
        padx = (dx // 2, dx - (dx // 2))
    if dy > 0:
        pady = (dy // 2, dy - (dy // 2))
    img = np.pad(img, (pady, padx, (0, 0)), "constant", constant_values=0)

    return img
