Note that raw data files providing pre-computed optical responses for different nanolibraries can be included in this folder (e.g. .mat files)
Any file format may be provided since the libraryClass.py classes will incorporate corresponding calls to unpack the data when needed for training


PROPER USE FOR LIBRARY CLASSES:
- parameters of length are/should be provided in standard units of m
- parameters of angle are/should be provided in standard units of radians (ideally wrapped to 2pi but not necessary)
- transmission values are/should be unitless and between 0 and 1 
- it is good practice to make wavelength the last dimension since that dimension is frequently reduced in many cases


NOTES ON INCLUDED DATA FILES:

o data_Nanofins_Unit350nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on september 2021 (Dean Hazineh)
    - 350 nm unit cell size
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 5 nm step size in fin dimensions between 60 to 300 nm 
    - 
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~500um above structure 
    - Transmission manually computed by calculating x and y polarized light energy at single far away point ~500um above structure
        and normalizing by energy at that point with no structure present. 
        (The total light transmitted (polarization averaged) by this method is compared against the averaged Lumerical power monitor and shows good agreement between the two methods)


o data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on march 2022 (Dean Hazineh)
    - 180 nm unit cell size
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 1 nm step size in cylinder radius between 30 and 150 nm
    -
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~um above structure
        - phasex and phasey are equivalent so just phasex is uploaded in the mat file 
    - Transmission computed by Lumerical power monitor above nanostructure (lumerical built in call automatically normalizes to source power without any user work)

o data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat:
    - Generated via Lumerical FDTD parameter sweep on July 2022 (Dean Hazineh)
    - 350 nm unit cell seize
    - response vs 1 nm step size in incident wavelength from 310 to 750 nm
    - response vs 1 nm step size in cylinder radius between 60 and 300 nm
    -
    - X and Y polarized plane wave sources simultaneously incident (injected inside glass susbtrate)
    - Bloch boundary conditions transverse to unit cell and PML normal
    - Record phase (minus reference field consisting of no TiO2 structure) at single point ~500um above structure 
    - Transmission manually computed by calculating x and y polarized light energy at single far away point ~500um above structure
        and normalizing by energy at that point with no structure present. 
        (The total light transmitted (polarization averaged) by this method is compared against the averaged Lumerical power monitor and shows good agreement between the two methods)

o data_Nanofins_Unit350nm_Height600nm_RCWATF_9b.pickle
    - Generated using dflat/cell_library_generation/generate_cell_library.py with the rcwa_tf engine on Aug 2022 (Dean Hazineh)
    - cell_fun=lib_gen.assemble_ER_rectangular_fin in DFlat version 1.0.1
    - 350 nm unit cell size
    - response vs 5 nm step size in wavelength from 400 to 705 nm
    - response vs 5 nm step size in fin lengths along x and y between 60 and 295 nm

    - rcwa_settings = {
        "wavelength_set_m": wavelength_set_m,
        "thetas": [0.0 for i in wavelength_set_m],
        "phis": [0.0 for i in wavelength_set_m],
        "pte": [1.0 for i in wavelength_set_m],
        "ptm": [1.0 for i in wavelength_set_m],
        "pixelsX": 1,
        "pixelsY": 1,
        "PQ": [FM, FM],
        "Lx": 350e-9,
        "Ly": 350e-9,
        "L": [1e-3, 600.0e-9, 1e-3],
        "Lay_mat": ["SiO2", "Vacuum", "Vacuum"],
        "material_dielectric": "TiO2",
        "er1": "Vacuum",
        "er2": "Vacuum",
        "Nx": 512,
        "Ny": 512,
        "parameterization_type": "None",
        "batch_wavelength_dim": False,  
        "dtype": tf.float64,
        "cdtype": tf.complex128,
    }

    - Importantly, this simulation is different from the FDTD runs. Propagation from air -> 1 mm SiO2 Substrate -> air embedding is considered. No PML conditions are used
    - This should be more consistent to the transmission spectra obtained in real life.
    
