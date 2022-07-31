import tensorflow as tf
from dflat.datasets_metasurface_cells.libraryClass import *


def optical_response_to_param(ms_trans_asList, ms_phase_asList, wavelength_m_asList, libraryName, reshape=False):
    """Converts a list of metasurface transmission and phase profiles to a list of assembled metasurface shape profiles
    (spatial distributions of nano-shape parameters) corresponding to the physical realization, at the specified
    wavelength channel. These metasurfaces are assemled from structures in one of the included raw FDTD libraries.

    Args:
        `ms_trans_asList` (list): List of metasurface transmission profiles. Each transmission profile must be of shape
            (m, Ny, Nx), where m==2 when a polarization sensitive library (e.g. nanofins) is used and m==1 when a
            polarization insensitive library (e.g. nanocylinders) is used.
        `ms_phase_asList` (list): List of metasurface phase profiles, same requirements as the transmission profile input.
        `wavelength_m_asList` (list): List of wavelength channels for which each metasurface phase and transmission
            profile is to be realized on.
        `libraryName` (str): Name of the library to assemble the metasurface from. Options include
            "Nanofins_U350nm_H600nm", "Nanocylinders_U180nm_H600nm". See neural_cell_model/core/libraryClass.py for
            more details.
        `reshape` (bool): When reshape is True, the shape vector is spatially reshaped to match the inputted
            transmission and phase profile input, i.e. (DoP, Ny, Nx) where DoP is the degree of the cell shape.
            When False, the vector is returned in flattened form as (Ny*Nx, DoP).

    Returns:
        `list`: Shape vector with nanostructure dimensions in units of m.
        `list`: Shape vector normalized by mlp bounds to [0,1]


    """
    # Get the meta-atom library
    library = loadLibrary(libraryName)
    shape_Vector, shape_Vector_norm = library.optical_response_to_param(
        ms_trans_asList, ms_phase_asList, wavelength_m_asList, reshape
    )

    return shape_Vector, shape_Vector_norm


def loadLibrary(libraryName):
    # libraryName is passed in as a string
    if libraryName not in listLibraryNames:
        raise ValueError("loadLibrary: requested meta-library is not one of the supported options")
    else:
        library = globals()[libraryName]
        library = library()

    return library
