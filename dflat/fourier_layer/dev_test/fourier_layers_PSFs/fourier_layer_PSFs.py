import dflat.data_structure as df_struct
import dflat.fourier_layer as df_fourier

import dflat.plot_utilities as df_plot
import dflat.tools as df_tools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def call_psf_layer_mono(engine):
    ### Define new propagation parameter
    savepath = "dflat/fourier_layer/dev_test/output/fourier_layers_PSFs/"
    propagation_settings = {
        # Mandatory Keys
        "wavelength_m": 532e-9,
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 4e-3,
        "initial_sensor_dx_m": {"x": 500e-9, "y": 500e-9},
        "sensor_pixel_size_m": {"x": 500e-9, "y": 500e-9},
        "sensor_pixel_number": {"x": 126, "y": 126},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
        "accurate_measurement": True,  # Flag ensures output grid is exact but is expensive
    }
    prop_params = df_struct.prop_params(propagation_settings, verbose=True)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    cidx = int((prop_params["sensor_pixel_number"]["x"]-1)/2)
    sensx, sensy = df_plot.get_detector_pixel_coordinates(prop_params)

    ### Do PSF Layer Calls
    zfocus = 50e-3
    zlocs = [30e-3, 50e-3, 80e-3]
    point_source_locs = np.array([[0, 0, zo] for zo in zlocs])

    # run 2D Calculation
    psf_layer = df_fourier.PSF_Layer_Mono(prop_params)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [zfocus], [{"x": 0, "y": 0}])
    psf_int, psf_phase = psf_layer([lens_trans, lens_phase], point_source_locs)

    # Run radial version Calculation
    psf_layer = df_fourier.PSF_Layer_Mono(prop_params_radial)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [zfocus], [{"x": 0, "y": 0}])
    psf_int_radial, psf_phase_radial = psf_layer([lens_trans, lens_phase], point_source_locs)

    # get airy disk profile
    psf_airy = df_tools.airy_disk(prop_params["wavelength_m"], prop_params["radius_m"], sensx, prop_params["sensor_distance_m"])[cidx, :]

    ###
    fig = plt.figure()
    ax = df_plot.addAxis(fig, 2, 3)
    for i in range(3):
        ax[i].imshow(psf_int[0, i])
        ax[i + 3].imshow(psf_int_radial[0, i])
    plt.savefig(savepath + "psf_layer_mono_" + engine + "_int.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 2, 3)
    for i in range(3):
        ax[i].imshow(psf_phase[0, i])
        ax[i + 3].imshow(psf_phase_radial[0, i])
    plt.savefig(savepath + "psf_layer_mono_" + engine + "_phase.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 1, 3)
    for i in range(3):
        ax[i].plot(psf_int[0, i, cidx, :], "kx-")
        ax[i].plot(psf_int_radial[0, i, cidx, :], "bx-")
    ax[1].plot(psf_airy, "r*-")
    plt.savefig(savepath + "psf_layer_mono_" + engine + "_diff_limited.png")

    plt.close()
    return


def call_psf_layer(engine):
    savepath = "dflat/fourier_layer/dev_test/output/fourier_layers_PSFs/"

    ### Define new propagation parameter
    propagation_settings = {
        # Mandatory Keys
        "wavelength_set_m": [450e-9, 532e-9, 700e-9],
        "ms_samplesM": {"x": 513, "y": 513},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": 256e-6,
        "sensor_distance_m": 6e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": False,
        "diffractionEngine": engine,
        # Optional Keys
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
        "accurate_measurement": True,  # Flag ensures output grid is exact but is expensive
    }
    prop_params = df_struct.prop_params(propagation_settings)
    propagation_settings["radial_symmetry"] = True
    prop_params_radial = df_struct.prop_params(propagation_settings)

    ### Do PSF Layer Calls
    zfocus = 50e-3
    zlocs = [30e-3, 50e-3, 80e-3]
    zoffset = [{"x": 0, "y": 0}]
    point_source_locs = np.array([[0, 0, zo] for zo in zlocs])

    # run 2D Calculation
    psf_layer = df_fourier.PSF_Layer(prop_params)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params, [532e-9], [zfocus], zoffset)
    psf_int, psf_phase = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=True)

    # Run radial version Calculation
    psf_layer = df_fourier.PSF_Layer(prop_params_radial)
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(prop_params_radial, [532e-9], [zfocus], zoffset)
    psf_int_radial, psf_phase_radial = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=True)

    ###
    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(psf_int[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "psf_layer_" + engine + "int.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(psf_phase[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "psf_layer_" + engine + "phase.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(psf_int_radial[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "psf_layer_" + engine + "int_radial.png")

    fig = plt.figure()
    ax = df_plot.addAxis(fig, 3, 3)
    pltcounter = 0
    for i in range(3):
        for j in range(3):
            ax[pltcounter].imshow(psf_phase_radial[i, 0, j, :, :])
            pltcounter += 1
    plt.savefig(savepath + "psf_layer_" + engine + "phase_radial.png")

    plt.close()
    return


if __name__ == "__main__":
    ### Check the mono psf layer for PSF
    call_psf_layer_mono("fresnel_fourier")
    call_psf_layer_mono("ASM_fourier")

    # Check the standard psf layer for PSF
    call_psf_layer("fresnel_fourier")
    call_psf_layer("ASM_fourier")
    