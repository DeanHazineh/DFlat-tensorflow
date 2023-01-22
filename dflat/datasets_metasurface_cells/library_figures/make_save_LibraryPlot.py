import dflat.datasets_metasurface_cells as df_lib

if __name__ == "__main__":
    ### Plot the nanofins
    # fdtd_nanofins_350 = df_lib.Nanofins_U350nm_H600nm()
    # fdtd_nanofins_350.plotLibrary(savepath="dflat/datasets_metasurface_cells/output/fdtd_nanofins_350")

    ### Plot the nanocylinders
    # fdtd_nanocylinder_180 = df_lib.Nanocylinders_U180nm_H600nm()
    # fdtd_nanocylinder_180.plotLibrary(savepath="dflat/datasets_metasurface_cells/output/fdtd_nanocylinder_180")

    ### Plot the ellipse library
    fdtd_nanoellipse_350 = df_lib.Nanoellipse_U350nm_H600nm()
    fdtd_nanoellipse_350.plotLibrary(savepath="dflat/datasets_metasurface_cells/output/fdtd_nanoellipse_350")
