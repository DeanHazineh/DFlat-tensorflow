import dflat.datasets_metasurface_cells.libraryClass as df_dataLibrary
import pickle

if __name__ == "__main__":

    view_library = df_dataLibrary.Nanofins_U350nm_H600nm_RCWA()
    view_library.plotLibrary(idx=28)

    # view_library = df_dataLibrary.Nanoellipse_U350nm_H600nm()
