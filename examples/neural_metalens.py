import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import dflat.optimization_helpers as df_optimizer
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural
import dflat.data_structure as df_struct
import dflat.plot_utilities as plt_util

# This import is needed if we want to use a cell library for table-lookup to initialize a particular starting lens
# import dflat.datasets_metasurface_cells as df_library


class pipeline_Metalens_MLP(df_optimizer.Pipeline_Object):
    def __init__(self, propagation_parameters, point_source_locs, savepath, saveAtEpochs=None):
        super(pipeline_Metalens_MLP, self).__init__(savepath, saveAtEpochs)

        # Add inputs to class attributes
        self.propagation_parameters = propagation_parameters
        self.point_source_locs = point_source_locs

        # define computational layers
        mlp_model = "MLP_Nanocylinders_Dense64_U180_H600"
        self.mlp_latent_layer = df_neural.MLP_Latent_Layer(mlp_model)
        self.psf_layer = df_fourier.PSF_Layer_Mono(propagation_parameters)  # Note the other psf layers one can use

        # Define initial starting condition for the metasurface latent tensor
        init_latent_tensor = tf.zeros(propagation_parameters["grid_shape"], dtype=tf.float64)
        self.latent_tensor_variable = tf.Variable(init_latent_tensor, trainable=True, dtype=tf.float64)

        ## The lens could be initialized in another state like a focusing lens:
        # focus_trans, focus_phase, _, _ = df_fourier.focus_lens_init(propagation_parameters, [532e-9], [0.3], [{"x": 0, "y": 0}])
        # _, norm_param = df_library.optical_response_to_param([focus_trans], [focus_phase], [532e-9], "Nanocylinders_U180nm_H600nm", reshape=True)
        # init_latent_tensor = df_tools.param_to_latent(init_norm_param)
        # self.latent_tensor_variable = tf.Variable(init_latent_tensor, trainable=True, dtype=tf.float64)

        return

    def __call__(self):
        # Compute the PSF
        wavelength_m = self.propagation_parameters["wavelength_m"]
        out = self.mlp_latent_layer(self.latent_tensor_variable, [wavelength_m])
        psf_intensity, psf_phase = self.psf_layer(out, self.point_source_locs)

        # Save the last lens and psf for plotting later
        self.last_lens = out
        self.last_psf = psf_intensity

        return psf_intensity

    def visualizeTrainingCheckpoint(self, saveto):
        # This overrides the baseclass visualization call function, called during checkpoints
        savefigpath = self.savepath + "/trainingOutput/"
        radial_flag = self.propagation_parameters["radial_symmetry"]

        # Helper call that returns simple definition of cartesian axis on lens and output space (mm)
        xl, yl = plt_util.get_lens_pixel_coordinates(self.propagation_parameters)
        xd, yd = plt_util.get_detector_pixel_coordinates(self.propagation_parameters)
        xl, yl = xl * 1e3, yl * 1e3
        xd, yd = xd * 1e3, yd * 1e3

        # Plot Grab some items we want to visualize
        # Latent tensor can be converted back to the physical shape dimensions
        latent_tensor_state = self.latent_tensor_variable.numpy()
        shapeVector = np.squeeze(self.mlp_latent_layer.latent_to_unnorm_shape(latent_tensor_state))
        trans = np.squeeze(self.last_lens[0])
        phase = np.squeeze(self.last_lens[1])

        fig = plt.figure(figsize=(40, 10))
        axList = plt_util.addAxis(fig, 1, 3)
        lens_extent = (min(xl), max(xl), max(yl), min(yl))
        det_extent = (min(xd), max(xd), max(yd), min(yd))
        if radial_flag:
            axList[0].plot(xl, shapeVector * 1e9)
            plt_util.formatPlots(fig, axList[0], None, xlabel="lens x (mm)", ylabel="radius length (nm)", setAspect="auto")
            axList[1].plot(xl, phase, "bx-")
            axList[1].plot(xl, trans, "kx-")
            plt_util.formatPlots(fig, axList[1], None, xlabel="lens x (mm)", ylabel="Trans and Phase", setAspect="auto")
        else:
            im0 = axList[0].imshow(shapeVector * 1e9, extent=lens_extent)
            plt_util.formatPlots(
                fig,
                axList[0],
                im0,
                xlabel="lens x (mm)",
                ylabel="lens y (nm)",
                title="Learned Metasurface",
                setAspect="equal",
                addcolorbar=True,
                cbarTitle="Radius (nm)",
            )

            im1 = axList[1].imshow(phase, extent=lens_extent, vmin=-np.pi, vmax=np.pi)
            plt_util.formatPlots(
                fig,
                axList[1],
                im1,
                xlabel="lens x (mm)",
                ylabel="lens y (mm)",
                title="Learned Phase",
                setAspect="equal",
                addcolorbar=True,
                cbarTitle="Phase (radians)",
            )

        # Plot the recent PSF
        im = axList[2].imshow(self.last_psf[0, 0], extent=det_extent)
        plt_util.formatPlots(
            fig, axList[2], im, xlabel="det x (mm)", ylabel="det y (mm)", title="PSF Intensity", addcolorbar=True, setAspect="equal"
        )

        plt.savefig(savefigpath + "png_images/" + saveto + "epoch_checkpointFig.png")
        plt.savefig(savefigpath + "pdf_images/" + saveto + "epoch_checkpointFig.pdf")
        plt.close()
        return


def optimize_metalens_mlp(radial_symmetry, try_gpu=True):
    savepath = "examples/output/metalens_example_radial" + str(radial_symmetry) + "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Define propagation parameters for psf calculation
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_m": 532e-9,  # wavelength_set_m would be used if the PSF_layer was used instead of psf_layer_mono
            "ms_samplesM": {"x": 555, "y": 555},
            "ms_dx_m": {"x": 10 * 180e-9, "y": 10 * 180e-9},
            "radius_m": 1.0e-3 / 2.0,
            "sensor_distance_m": 10e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 501, "y": 501},
            "radial_symmetry": radial_symmetry,
            "diffractionEngine": "fresnel_fourier",
            ### Optional keys
            "automatic_upsample": False,  # If true, it will try to automatically determine good upsample factor for calculations
            "manual_upsample_factor": 1,  # Otherwise you can manually dictate upsample factor
        },
        verbose=True,
    )

    # Point_source locs we want to compute the psf for
    point_source_locs = np.array([[0.0, 0.0, 1e6]])  # on-axis ps at 1e6 m away (~infinity)

    # Call the pipeline
    pipeline = pipeline_Metalens_MLP(propagation_parameters, point_source_locs, savepath, saveAtEpochs=5)
    # pipeline.customLoad()  # restore previous training checkpoint if it exists

    # Define custom Loss function (Should always have pipeline_output as the function input if you use the helper)
    # Otherwise you can easily write your own train loop for more control
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    cidx_y = sensor_pixel_number["y"] // 2
    cidx_x = sensor_pixel_number["x"] // 2

    def loss_fn(pipeline_output):
        return -pipeline_output[0, 0, cidx_y, cidx_x]

    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    df_optimizer.run_pipeline_optimization(pipeline, optimizer, num_epochs=100, loss_fn=tf.function(loss_fn), allow_gpu=try_gpu)

    return


if __name__ == "__main__":
    # Play around with the settings in the function call to compare gpu vs no gpu, different propagators, lr, etc.
    optimize_metalens_mlp(radial_symmetry=True, try_gpu=True)
