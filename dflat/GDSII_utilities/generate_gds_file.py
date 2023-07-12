import numpy as np
import gdspy
import cv2
import time
from tqdm.auto import tqdm


def upsample_with_cv2(inputs, upsample_factor):
    # Each input should have the shape (D, Ny, Nx)
    outputs = []
    for input in inputs:
        this = cv2.resize(np.transpose(input, [1, 2, 0]), None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_AREA)
        # cv2 resize squeezes one dimension so we should put that back
        if len(this.shape) != 3:
            this = np.expand_dims(this, -1)
        outputs.append(np.transpose(this, [2, 0, 1]))

    return outputs


def assemble_nanofin_gdsII(shape_array, rotation_array, cell_size, savepath, boolean_mask=None, gds_unit=1e-6, gds_precision=1e-9, tag=None, add_markers=True):
    # If no boolean mask is provided, then set to True everywhere
    if boolean_mask is None:
        boolean_mask = np.ones_like(shape_array, dtype=bool)

    ### Validate input shapes
    if len(shape_array.shape) != 3 or len(rotation_array.shape) != 3 or len(boolean_mask.shape) != 3:
        raise ValueError("Shape, rotation, and mask array should be rank 3 array")

    if shape_array.shape[0] != 2:
        raise ValueError("shape_array should have 2 for the first dimension (length x and length y)")

    if shape_array.shape[1:] != boolean_mask.shape[1:]:
        raise ValueError("shape_array and boolean_mask should have the same last two dimensions")

    print("Writing metasurface shapes to GDS File")
    start = time.time()

    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    cell = lib.new_cell("MAIN")

    for yi in tqdm(range(shape_array.shape[-2])):
        for xi in range(shape_array.shape[-1]):
            if boolean_mask[0, yi, xi]:
                xoffset = cell_size * (xi + 0.5) / gds_unit
                yoffset = cell_size * (yi + 0.5) / gds_unit

                Lx = shape_array[0, yi, xi] / gds_unit
                Ly = shape_array[1, yi, xi] / gds_unit

                rect = gdspy.Rectangle((-Lx / 2, -Ly / 2), (Lx / 2, Ly / 2))
                rect.rotate(rotation_array[0, yi, xi])
                rect.translate(xoffset, yoffset)
                cell.add(rect)

    ### Add some lens markers (bottom and left)
    if add_markers:
        halfx = shape_array.shape[-1] // 2
        halfy = shape_array.shape[-2] // 2
        marker_span = 80e-6 / gds_unit

        x_loc = cell_size * (halfx + 0.5) / gds_unit
        htext = gdspy.Text("+", marker_span, (x_loc, -3 * marker_span))
        cell.add(htext)
        htext = gdspy.Text("+", marker_span, (x_loc, -6 * marker_span))
        cell.add(htext)

        y_loc = cell_size * (halfy + 0.5) / gds_unit
        htext = gdspy.Text("+", marker_span, (-3 * marker_span, y_loc))
        cell.add(htext)
        htext = gdspy.Text("+", marker_span, (-6 * marker_span, y_loc))
        cell.add(htext)

    ### Add some text above the lens
    if tag is not None:
        text_height = 5 * cell_size / gds_unit
        y_loc = cell_size * (shape_array.shape[-2] + 0.5) / gds_unit + 2 * text_height
        htext = gdspy.Text("DFlat V2 GDSII", text_height, (x_loc, y_loc))
        cell.add(htext)

    lib.write_gds(savepath + ".gds")
    mystyle = {(1, 0): {"fill": "#000000", "stroke": "#CC00FF"}}
    # cell.write_svg(savepath + ".svg", scaling=gds_unit / gds_precision, style=mystyle, background="#FFFFFF")

    ###
    end = time.time()
    print("Completed writing and saving metasurface GDS File: Time: ", end - start)

    return
