import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import dflat.datasets_image as df_images
import dflat.data_structure as df_struct
import dflat.optimization_helpers as df_opt
import dflat.plot_utilities as df_plot
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural


class generate_hologram_metasurface(df_opt.Pipeline_Object):
    def __init__(self, target_int, prop_params, savepath, saveAtEpochs):
        super(generate_hologram_metasurface, self).__init__(savepath, saveAtEpochs)

        # Initialize the compuational pipeline
        self.target_int = tf.convert_to_tensor(target_int / np.sum(target_int, axis=(1, 2), keepdims=True), dtype=tf.float64)
        self.propagation_parameters = prop_params
        self.mlp_latent_layer = df_neural.MLP_Latent_Layer("MLP_Nanofins_Dense512_U350_H600")
        self.propagator = df_fourier.Propagate_Planes_Layer_Mono(prop_params)

        # initialize trainable lens
        ms_samplesM = prop_params["ms_samplesM"]
        init_latent_tensor = tf.zeros((2, ms_samplesM["y"], ms_samplesM["x"]), dtype=tf.float64)
        self.latent_tensor_variable = tf.Variable(init_latent_tensor, trainable=True, dtype=tf.float64, name="metasurface_latent_tensor")

    def __call__(self):
        # propagate field to output plane
        wavelength_m = self.propagation_parameters["wavelength_m"]
        out = self.mlp_latent_layer(self.latent_tensor_variable, [wavelength_m])
        norm = tf.math.reduce_sum(out[0][0] ** 2, axis=(1, 2), keepdims=True)

        field_ampl, _ = self.propagator(out)
        field_int = field_ampl**2 / norm

        # Add the field int to class attribute
        self.field_int = field_int

        # Compute the L1 loss of target vs realized hologram intensity
        error = tf.math.reduce_sum(tf.math.abs(field_int - self.target_int))

        return error

    def visualizeTrainingCheckpoint(self, epoch_str):

        energy_transmitted = np.sum(self.field_int, axis=(1, 2))
        fig = plt.figure()
        ax = df_plot.addAxis(fig, 2, 2)
        ax[0].imshow(self.field_int[0])
        ax[0].set_title("x-Polarized light")
        # ax[0].set_title(f"{energy_transmitted[0]:2.2f}")
        ax[1].imshow(self.field_int[1])
        ax[1].set_title("y-Poalrized light")
        # ax[1].set_title(f"{energy_transmitted[1]:2.2f}")
        ax[2].imshow(self.target_int[0])
        ax[2].set_title("Target Hologram x")
        ax[3].imshow(self.target_int[1])
        ax[3].set_title("Target Hologram y")
        plt.savefig(self.savepath + "/trainingOutput/png_images/checkpoint_img_" + epoch_str + ".png")

        return


### Define propagation parameters and initial lens
propagation_parameters = df_struct.prop_params(
    {
        "wavelength_m": 532e-9,
        "ms_samplesM": {"x": 512, "y": 512},
        "ms_dx_m": {"x": 1e-6, "y": 1e-6},
        "radius_m": None,
        "sensor_distance_m": 5e-3,
        "initial_sensor_dx_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_size_m": {"x": 1e-6, "y": 1e-6},
        "sensor_pixel_number": {"x": 1024, "y": 1024},
        "radial_symmetry": False,
        "diffractionEngine": "fresnel_fourier",
        ###
        "accurate_measurement": True,
        "automatic_upsample": False,
        "manual_upsample_factor": 1,
    },
    verbose=True,
)


### Load an image and then threshold it to binary
sensor_dim = propagation_parameters["sensor_pixel_number"]
image_x = df_images.get_grayscale_image("githublogo.png", sensor_dim, resize_method="pad")
image_y = df_images.get_grayscale_image("text2image_MetaOptics.png", sensor_dim, resize_method="crop")
image_target = np.transpose(np.concatenate((image_x, image_y), axis=-1), [2, 0, 1])

thresh = 40
image_target[np.where(image_target < thresh)] = 0.0
image_target[np.where(image_target >= 10)] = 1.0

# fig = plt.figure()
# ax = df_plot.addAxis(fig, 1, 2)
# ax[0].imshow(image_target[0], cmap="gray")
# ax[1].imshow(image_target[1], cmap="gray")
# plt.show()

### Create hologram optimizer
savepath = "examples/output/dual_polarization_hologram/"
saveAtEpoch = 5
pipeline = generate_hologram_metasurface(image_target, propagation_parameters, savepath, saveAtEpoch)
# pipeline.customLoad()

optimizer = tf.keras.optimizers.Adam(1e-1)
df_opt.run_pipeline_optimization(pipeline, optimizer, num_epochs=200, loss_fn=None, allow_gpu=True)
