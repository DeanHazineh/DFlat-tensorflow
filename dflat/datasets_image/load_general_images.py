from pathlib import Path
from .core.load_image_fun import *


def get_path_to_data(file_name: str):
    ## This is not the accepted solution but it should work for bootstrapping research with few users
    resource_path = Path(__file__).parent / "datasets"
    return resource_path.joinpath(file_name)


def get_grayscale_image(img_name, img_dim=None, resize_method="crop"):
    """Return image as grayscale image, resized to img_dim while preserving the original aspect ratio.

    Args:
        `img_name`: Filename in path datasets/grayscale_images/x.png
        `img_dim` (dict, optional): Dictionary with key "x" and "y", denoting number of pixels along each direction
        `resize_method` (str, optional): Either preserve aspect ratio by "crop" or by "pad"

    Returns:
        float: image
    """
    return load_png_as_grayscale(get_path_to_data("grayscale_images/" + img_name), img_dim, resize_method)
