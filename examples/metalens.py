import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dflat.optimization_helpers as df_optimizer
import dflat.fourier_layer as df_fourier_layer
import dflat.neural_optical_layer as df_neural_layer
import dflat.physical_optical_layer as df_physical_layer
import dflat.data_structure as df_data_struct
import dflat.tools as df_tools


class pipeline_Metalens_MLP(df_optimizer.Pipeline_Object):
    def __init__(self, propagation_parameters, point_source_locs, savepath, saveAtEpochs=None):

        super(pipeline_Metalens_MLP, self).__init__(savepath, saveAtEpochs)
        self.propagation_parameters = propagation_parameters
        self.point_source_locs = point_source_locs

        # define computational layers
        mlp_model = "MLP_Nanocylinders_Dense64_U180_H600"
        self.mlp_latent_layer = df_neural_layer.MLP_Latent_Layer(mlp_model)
        self.psf_layer = df_fourier_layer.PSF_Layer_Mono(propagation_parameters)

        # Make uniform state latent tensor as initial variable for metasurface with helper function
        gridShape = propagation_parameters["grid_shape"]
        latent_tensor_variable = self.mlp_latent_layer.initialize_input_tensor("uniform", gridShape, tf.float64)
        self.latent_tensor_variable = tf.Variable(
            latent_tensor_variable, trainable=True, dtype=tf.float64, name="metasurface_latent_tensor"
        )

        # # The lens can be initialized in another state like a focusing lens
        # focus_trans, focus_phase, _, _ = df_fourier_layer.focus_lens_init(
        #     propagation_parameters, [532e-9], [0.5], [{"x": 0, "y": 0}]
        # )
        # _, norm_shape = df_neural_layer.optical_response_to_param(
        #     [focus_trans], [focus_phase], [532e-9], "Nanocylinders_U180nm_H600nm", reshape=True
        # )
        # latent_tensor = df_tools.param_to_latent(norm_shape[0])
        # latent_tensor_variable = tf.Variable(
        #     latent_tensor, trainable=True, dtype=tf.float64, name="metasurface_latent_tensor"
        # )

    def __call__(self):
        wavelength_m = self.propagation_parameters["wavelength_m"]
        out = self.mlp_latent_layer(self.latent_tensor_variable, [wavelength_m])
        psf_intensity, psf_phase = self.psf_layer(out, self.point_source_locs)

        # move last PSF calculation to the class variable for viewing later
        self.last_psf = psf_intensity

        return psf_intensity

    def visualizeTrainingCheckpoint(self, saveto):
        # This overrides the baseclass visualization call function, called during checkpoints
        savefigpath = self.savepath + "/trainingOutput/"
        radial_flag = self.propagation_parameters["radial_symmetry"]

        # Get variables to plot
        latent_tensor_state = self.get_variable_by_name("metasurface_latent_tensor")[0]
        norm_shape = df_tools.latent_to_param(latent_tensor_state)
        lens_phase, lens_trans = self.mlp_latent_layer(
            self.latent_tensor_variable, [self.propagation_parameters["wavelength_m"]]
        )
        xvect, yvect = df_fourier_layer.getCoordinates_vector(
            self.propagation_parameters["ms_samplesM"],
            self.propagation_parameters["ms_dx_m"],
            radial_flag,
            tf.float32,
        )
        xvect_det, yvect_det = df_fourier_layer.getCoordinates_vector(
            self.propagation_parameters["sensor_pixel_number"],
            self.propagation_parameters["sensor_pixel_size_m"],
            False,
            tf.float32,
        )

        # Make a visualization plot
        fig = plt.figure(figsize=(20, 10))
        axList = df_tools.addAxis(fig, 1, 3)
        if radial_flag:
            axList[0].plot(xvect[0, :] * 1e3, norm_shape[0, 0, :])
            axList[1].plot(xvect[0, :] * 1e3, lens_phase[0, 0, 0, :], "bx-")
            axList[1].plot(xvect[0, :] * 1e3, lens_trans[0, 0, 0, :], "kx-")
            df_tools.formatPlots(
                fig,
                axList[0],
                None,
                xlabel="lens x (mm)",
                ylabel="normalized shape",
                title="Trained Metasurface",
                setAspect="auto",
            )
            df_tools.formatPlots(
                fig,
                axList[1],
                None,
                xlabel="lens x (mm)",
                ylabel="Trans (black) | Phase (blue)",
                title="Metasurface Trans and Phase",
                setAspect="auto",
            )

        else:
            im0 = axList[0].imshow(
                norm_shape[0, :, :], extent=(np.min(xvect), np.max(xvect), np.max(yvect), np.min(yvect))
            )
            im1 = axList[1].imshow(
                lens_phase[0, 0, :, :], extent=(np.min(xvect), np.max(xvect), np.max(yvect), np.min(yvect))
            )
            df_tools.formatPlots(
                fig,
                axList[0],
                im0,
                xlabel="lens x (mm)",
                ylabel="lens y (nm)",
                title="Trained Metasurface",
                setAspect="equal",
                addcolorbar=True,
                cbartitle="Normalized Radius (nm)",
            )
            df_tools.formatPlots(
                fig,
                axList[1],
                im1,
                xlabel="lens x (mm)",
                ylabel="lens y (mm)",
                title="Metasurface Phase",
                setAspect="equal",
                addcolorbar=True,
                cbartitle="Phase (radians)",
            )

        im = axList[2].imshow(
            self.last_psf[0, 0, 0, :, :],
            extent=(
                np.min(xvect_det) * 1e3,
                np.max(xvect_det) * 1e3,
                np.max(yvect_det) * 1e3,
                np.min(yvect_det) * 1e3,
            ),
        )
        df_tools.formatPlots(
            fig,
            axList[2],
            im,
            xlabel="det x (mm)",
            ylabel="det y (mm)",
            title="PSF Intensity",
            addcolorbar=True,
            setAspect="equal",
        )

        plt.savefig(savefigpath + "png_images/" + saveto + "epoch_checkpointFig.png")
        plt.savefig(savefigpath + "pdf_images/" + saveto + "epoch_checkpointFig.pdf")

        return


def optimize_metalens_mlp():

    # Define propagation parameters for psf calculation
    propagation_parameters = df_data_struct.prop_params(
        {
            "wavelength_m": 532e-9,
            "ms_length_m": {"x": 1.0e-3, "y": 1.0e-3},
            "ms_dx_m": {"x": 10 * 180e-9, "y": 10 * 180e-9},
            "radius_m": 1.0e-3 / 2.01,
            "sensor_distance_m": 10e-3,
            "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
            "sensor_pixel_number": {"x": 501, "y": 501},
            "radial_symmetry": True,
            "diffractionEngine": "fresnel_fourier",
            "accurate_measurement": True,  # Flag ensures output grid is exact but is expensive
        },
        verbose=True,
    )

    # Point_source locs we want to compute the psf for
    point_source_locs = np.array([[0.0, 0.0, 1e6]])  # on-axis ps at 1e6 m away (~infinity)

    # Call the pipeline
    savepath = "examples/output/metalens_example/"
    pipeline = pipeline_Metalens_MLP(propagation_parameters, point_source_locs, savepath, saveAtEpochs=5)
    pipeline.customLoad()

    # Define custom Loss function (Should always have pipeline_output as the function input)
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    cidx_y = sensor_pixel_number["y"] // 2
    cidx_x = sensor_pixel_number["x"] // 2

    def loss_fn(pipeline_output):
        return -pipeline_output[0, 0, 0, cidx_y, cidx_x]

    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    df_optimizer.run_pipeline_optimization(
        pipeline, optimizer, num_epochs=30, loss_fn=tf.function(loss_fn), allow_gpu=True
    )

    return


if __name__ == "__main__":
    optimize_metalens_mlp()
