import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import scipy.io as sio

import dflat.fourier_layer as df_fourier
import dflat.data_structure as df_struct
import dflat.tools.graphFunc as gF
from dflat.tools.diff_limited_psf import airy_disk

savepath = "dflat/unit_tests/deeper_testing_code/output/"


def diff_limited_psfs_PSF_Layer(inputs):
    radial_symmetry = inputs[0]
    engine = inputs[1]
    batch_loop = inputs[2]

    ### Define Simulation parameters
    simulationSettings = {
        "wavelength_set_m": [450e-9, 550e-9, 650e-9],
        "ms_length_m": {"x": 100e-6, "y": 100e-6},
        "ms_dx_m": {"x": 350e-9, "y": 350e-9},
        "radius_m": 100e-6 / 2.01,
        "sensor_distance_m": 200e-6,
        "initial_sensor_dx_m": {"x": 100e-9, "y": 100e-9},
        "sensor_pixel_size_m": {"x": 100e-9, "y": 100e-9},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": radial_symmetry,
        "diffractionEngine": engine,
        "accurate_measurement": True,
        "ASM_Pad_opt": 1.0,
    }
    parameters = df_struct.prop_params(simulationSettings, verbose=False)
    wavelength_set_m = parameters["wavelength_set_m"]
    num_wl = len(wavelength_set_m)

    ### Initialize a lens
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(
        parameters, wavelength_set_m, [1e6 for i in range(num_wl)], [{"x": 0, "y": 0} for i in range(num_wl)]
    )
    point_source_locs = np.array([[0.0, 0.0, 1e6]])

    ### Call the PSF layer
    psf_layer = df_fourier.PSF_Layer(parameters)
    ipsf, _ = psf_layer([lens_trans, lens_phase], point_source_locs, batch_loop=batch_loop)

    ### Make Plots of the computed PSF
    # define the sensor coordinates
    sensor_pixel_number = parameters["sensor_pixel_number"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    cidx_x = int(sensor_pixel_number["x"] // 2)
    cidx_y = int(sensor_pixel_number["y"] // 2)
    xvec = np.arange(sensor_pixel_number["x"])
    xvec = (xvec - cidx_x) * sensor_pixel_size_m["x"]

    fig = plt.figure(figsize=(20, 20))
    ax = gF.addAxis(fig, 3, 3)
    axcounter = 0
    for i in range(num_wl):
        for j in range(num_wl):
            ax[axcounter].imshow(ipsf[j, i, 0, :, :])
            gF.formatPlots(
                fig, ax[axcounter], None, rmvxLabel=True, rmvyLabel=True, title=f"{(wavelength_set_m[j] * 1e9):3.0f}"
            )
            axcounter = axcounter + 1
    plt.savefig(savepath + "PSF_2D_radialFlag" + str(radial_symmetry) + "_engine" + engine)
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = gF.addAxis(fig, 1, 3)
    axcounter = 0
    for i in range(num_wl):
        airy_prof = airy_disk(wavelength_set_m[i], parameters["radius_m"], xvec, parameters["sensor_distance_m"])
        ax[axcounter].plot(xvec, ipsf[i, i, 0, cidx_y, :] / ipsf[i, i, 0, cidx_y, cidx_x], "bo-")
        ax[axcounter].plot(xvec, airy_prof[cidx_y, :], "rx--")
        gF.formatPlots(
            fig,
            ax[axcounter],
            None,
            xlabel="sensor x",
            ylabel="Normalized Intensity",
            title="Wavelength (nm): " + f"{(wavelength_set_m[i] * 1e9):3.0f}",
        )
        axcounter = axcounter + 1
    plt.savefig(savepath + "PSF_slice_radialFlag" + str(radial_symmetry) + "_engine" + engine)
    plt.close()

    plt.show()

    return


def heart_singularity(inputs):
    ### Make a plot of the experimental measurements
    dataLoc = "dflat/unit_tests/deeper_testing_code/dataFiles/reprocessed_Daniel_experimentalData.mat"
    data = sio.loadmat(dataLoc)
    phases = data["phases"]
    intensity = data["intensity"]
    int_z = np.array(data["int_z"], dtype=float)
    num_sensor_distances = len(int_z)

    int_x = np.array(data["int_x"], dtype=float) * 1e6 - 36.5
    int_y = np.array(data["int_y"], dtype=float) * 1e6 - 24.5
    phase_x = np.array(data["phase_x"], dtype=float) * 1e6 - 36.5
    phase_y = np.array(data["phase_y"], dtype=float) * 1e6 - 24.5

    _, phase_cx_idx = min((val, idx) for (idx, val) in enumerate(np.abs(phase_x.flatten())))
    _, phase_cy_idx = min((val, idx) for (idx, val) in enumerate(np.abs(phase_y.flatten())))
    plotPhase = phases[:, :, :] - phases[phase_cy_idx, phase_cx_idx, :]
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    ### Generate Plot of Experimental Fields
    fig = plt.figure(figsize=(10, num_sensor_distances * 5))
    ax = gF.addAxis(fig, num_sensor_distances, 2)
    axcounter = 0
    for i in range(num_sensor_distances):

        plt_int = ax[axcounter].imshow(
            np.log10(intensity[:, :, i]),
            extent=(np.min(int_x), np.max(int_x), np.min(int_y), np.max(int_y)),
            origin="lower",
            cmap="viridis",
        )
        plt_pha = ax[axcounter + 1].imshow(
            plotPhase[:, :, i],
            extent=(np.min(phase_x), np.max(phase_x), np.min(phase_y), np.max(phase_y)),
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )

        gF.formatPlots(
            fig,
            ax[axcounter],
            plt_int,
            rmvxLabel=True,
            rmvyLabel=True,
            setxlim=[-24, 24],
            setylim=[-24, 24],
        )
        gF.formatPlots(
            fig,
            ax[axcounter + 1],
            plt_pha,
            rmvxLabel=True,
            rmvyLabel=True,
            setxlim=[-24, 24],
            setylim=[-24, 24],
        )

        axcounter = axcounter + 2
    plt.savefig(savepath + "HeartSingularity_Experiment")
    plt.close()

    ### Load in the validated heart singularity metasurface data and parameter file
    params_loc = "dflat/unit_tests/deeper_testing_code/dataFiles/metasurface_heart_parameters.pickle"
    with open(params_loc, "rb") as handle:
        exSimDict = pickle.load(handle)
    ms_phase = np.expand_dims(exSimDict["ms_phase_x"], 0)
    ms_trans = np.expand_dims(exSimDict["ms_trans"], 0)

    ### create a new fourier prop_params using the heart parameters file (matched to experimental data)
    engine = inputs[0]
    downsampleFactor = 5.0
    simulationSettings = {
        "wavelength_m": 532e-9,
        "ms_length_m": {"x": 0.8e-3, "y": 0.8e-3},
        "ms_dx_m": {"x": 8.0e-6, "y": 8.0e-6},
        "sensor_distance_m": 10e-3,  # placeholder to loop over values
        "initial_sensor_dx_m": {"x": 6.0e-8 * downsampleFactor, "y": 6.0e-8 * downsampleFactor},
        "sensor_pixel_size_m": {"x": 6.0e-8 * downsampleFactor, "y": 6.0e-8 * downsampleFactor},
        "sensor_pixel_number": {
            "x": int(1001 / downsampleFactor),
            "y": int(1001 / downsampleFactor),
        },
        "dtype": tf.float64,
        "radial_symmetry": False,
        "diffractionEngine": engine,
        "accurate_measurement": True,
        "ASM_Pad_opt": 1.0,
    }

    ### Propagate the fields, looping over sensor distances
    sensor_distance_m = [9.6e-3, 9.8e-3, 10.0e-3, 10.2e-3, 10.4e-3]
    ampl_stack = []
    phase_stack = []
    for dist in sensor_distance_m:
        print("Calculating: ", dist)
        simulationSettings["sensor_distance_m"] = dist
        parameters = df_struct.prop_params(simulationSettings, verbose=False)

        # Initialize the field propagator layer
        field_propagator = df_fourier.Propagate_Planes_Layer_Mono(parameters)
        out = field_propagator((ms_trans, ms_phase))
        ampl_stack.append(tf.expand_dims(out[0], 0))
        phase_stack.append(tf.expand_dims(out[1], 0))

    ampl_stack = tf.transpose(tf.stack(ampl_stack, axis=0), [0, 1, 2, 4, 3])
    phase_stack = tf.transpose(tf.stack(phase_stack, axis=0), [0, 1, 2, 4, 3])

    ### Generate Plot of Calculated Fields
    sensor_pixel_number = parameters["sensor_pixel_number"]
    sensor_pixel_size = parameters["sensor_pixel_size_m"]
    xvec = (np.arange(0, sensor_pixel_number["x"]) - sensor_pixel_number["r"]) * sensor_pixel_size["x"] * 1e6
    yvec = (np.arange(0, sensor_pixel_number["y"]) - sensor_pixel_number["r"]) * sensor_pixel_size["y"] * 1e6

    _, phase_cx_idx = min((val, idx) for (idx, val) in enumerate(np.abs(xvec.flatten())))
    _, phase_cy_idx = min((val, idx) for (idx, val) in enumerate(np.abs(yvec.flatten())))
    plotPhase = phase_stack[:, 0, 0, :, :] - tf.expand_dims(
        tf.expand_dims(phase_stack[:, 0, 0, phase_cy_idx, phase_cx_idx], 1), 1
    )
    plotPhase = np.arctan2(np.sin(plotPhase), np.cos(plotPhase))

    fig = plt.figure(figsize=(10, len(sensor_distance_m) * 5))
    ax = gF.addAxis(fig, len(sensor_distance_m), 2)
    axcounter = 0
    for i in range(len(sensor_distance_m)):
        plt_int = ax[axcounter].imshow(
            np.log10(ampl_stack[i, 0, 0, :, :] ** 2),
            extent=(np.min(xvec), np.max(xvec), np.min(yvec), np.max(yvec)),
            origin="lower",
            cmap="viridis",
        )
        plt_pha = ax[axcounter + 1].imshow(
            plotPhase[i, :, :],
            extent=(np.min(xvec), np.max(xvec), np.min(yvec), np.max(yvec)),
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
        )

        gF.formatPlots(
            fig,
            ax[axcounter],
            plt_int,
            rmvxLabel=True,
            rmvyLabel=True,
            setxlim=[-24, 24],
            setylim=[-24, 24],
        )
        gF.formatPlots(
            fig,
            ax[axcounter + 1],
            plt_pha,
            rmvxLabel=True,
            rmvyLabel=True,
            setxlim=[-24, 24],
            setylim=[-24, 24],
        )
        axcounter = axcounter + 2

    plt.savefig(savepath + "HeartSingularity_engine" + engine)
    plt.close()

    plt.show()

    return


def test_fast_ASM(inputs):
    radial_symmetry = inputs[0]

    ### Define Simulation parameters
    simulationSettings = {
        "wavelength_set_m": [450e-9, 550e-9, 650e-9],
        "ms_length_m": {"x": 100e-6, "y": 100e-6},
        "ms_dx_m": {"x": 350e-9, "y": 350e-9},
        "radius_m": 100e-6 / 2.01,
        "sensor_distance_m": 200e-6,
        "initial_sensor_dx_m": {"x": 100e-9, "y": 100e-9},
        "sensor_pixel_size_m": {"x": 100e-9, "y": 100e-9},
        "sensor_pixel_number": {"x": 256, "y": 256},
        "radial_symmetry": radial_symmetry,
        "diffractionEngine": "ASM_fourier",
        "accurate_measurement": True,
        "ASM_Pad_opt": 1.0,
    }
    parameters = df_struct.prop_params(simulationSettings, verbose=False)
    wavelength_set_m = parameters["wavelength_set_m"]
    num_wl = len(wavelength_set_m)

    ### Initialize a lens
    lens_trans, lens_phase, _, _ = df_fourier.focus_lens_init(
        parameters, wavelength_set_m, [1e6 for i in range(num_wl)], [{"x": 0, "y": 0} for i in range(num_wl)]
    )
    point_source_locs = np.array([[0.0, 0.0, 1e6]])

    ### Call the PSF layer
    psf_layer = df_fourier.PSF_Layer_MatrixBroadband(parameters)
    ipsf, _ = psf_layer([lens_trans, lens_phase], point_source_locs, wavelength_set_m)
    print(ipsf.shape)

    ### Make Plots of the computed PSF
    # define the sensor coordinates
    sensor_pixel_number = parameters["sensor_pixel_number"]
    sensor_pixel_size_m = parameters["sensor_pixel_size_m"]
    cidx_x = int(sensor_pixel_number["x"] // 2)
    cidx_y = int(sensor_pixel_number["y"] // 2)
    xvec = np.arange(sensor_pixel_number["x"])
    xvec = (xvec - cidx_x) * sensor_pixel_size_m["x"]

    fig = plt.figure(figsize=(20, 20))
    ax = gF.addAxis(fig, 3, 3)
    axcounter = 0
    for i in range(num_wl):
        for j in range(num_wl):
            ax[axcounter].imshow(ipsf[j, i, 0, :, :])
            gF.formatPlots(
                fig, ax[axcounter], None, rmvxLabel=True, rmvyLabel=True, title=f"{(wavelength_set_m[j] * 1e9):3.0f}"
            )
            axcounter = axcounter + 1
    plt.savefig(savepath + "MatrixASM_PSF_2D_radialFlag" + str(radial_symmetry))
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = gF.addAxis(fig, 1, 3)
    axcounter = 0
    for i in range(num_wl):
        airy_prof = airy_disk(wavelength_set_m[i], parameters["radius_m"], xvec, parameters["sensor_distance_m"])
        ax[axcounter].plot(xvec, ipsf[i, i, 0, cidx_y, :] / ipsf[i, i, 0, cidx_y, cidx_x], "bo-")
        ax[axcounter].plot(xvec, airy_prof[cidx_y, :], "rx--")
        gF.formatPlots(
            fig,
            ax[axcounter],
            None,
            xlabel="sensor x",
            ylabel="Normalized Intensity",
            title="Wavelength (nm): " + f"{(wavelength_set_m[i] * 1e9):3.0f}",
        )
        axcounter = axcounter + 1
    plt.savefig(savepath + "MatrixASM_PSF_slice_radialFlag" + str(radial_symmetry))
    plt.close()

    plt.show()

    return


def run_all_tests():
    fun = [
        # diff_limited_psfs_PSF_Layer,
        # diff_limited_psfs_PSF_Layer,
        # diff_limited_psfs_PSF_Layer,
        # diff_limited_psfs_PSF_Layer,
        # heart_singularity,
        # heart_singularity,
        test_fast_ASM,
        test_fast_ASM,
    ]
    arguments = [
        # [True, "fresnel_fourier", False],
        # [True, "ASM_fourier", False],
        # [False, "fresnel_fourier", True],
        # [False, "ASM_fourier", True],
        # ["fresnel_fourier"],
        # ["ASM_fourier"],
        [True],
        [False],
    ]

    for idx, call_test in enumerate(fun):
        # try:
        #     with tf.device("/gpu:0"):
        #         call_test(arguments[idx])
        #        except:
        # "GPU execution failed (probably due to memory)"
        with tf.device("/cpu:0"):
            call_test(arguments[idx])

    return


if __name__ == "__main__":
    run_all_tests()
