from dflat.datasets_metasurface_cells import *
from dflat.neural_optical_layer.mlp_initialization_utilities import loadLibrary


def test_open_meta_libraries():
    for lib_name in listLibraryNames:
        lib_class = loadLibrary(lib_name)

    return
