import tensorflow as tf
from .util_neural import leakyrelu100, gaussian_activation
from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600
from .arch_Core_class import GFF_Projection_layer, GFF_Projection_layer

mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600",
    # "MLP_Nanocylinders_Dense128_U180_H600",
    # "MLP_Nanocylinders_Dense64_U180_H600",
    "MLP_Nanofins_Dense1024_U350_H600",
    "MLP_Nanofins_Dense512_U350_H600",
    # "MLP_Nanofins_Dense256_U350_H600",
    # "MLP_Nanofins_Dense64_U350_H600",
]


## USABLE MLP MODELS
class MLP_Nanocylinders_Dense256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense256_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense128_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense128_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense128_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense128_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                128,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanocylinders_Dense64_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanocylinders_Dense64_U180_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense64_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense64_U180_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


##
class MLP_Nanofins_Dense1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense1024_U350_H600, self).__init__(dtype)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                1024,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                1024,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense512_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense512_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense512_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense512_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                512,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                512,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense256_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense256_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense256_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense256_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                256,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense64_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(MLP_Nanofins_Dense64_U350_H600, self).__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense64_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense64_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


### TESTING


# class MLP_Nanofins_2GDense1024_U350_H600(MLP_Nanofins_U350_H600):
#     def __init__(self, a=0.5, dtype=tf.float64):
#         super(MLP_Nanofins_2GDense1024_U350_H600, self).__init__(dtype)
#         # a is a hyper-parameter that controlls the gaussian activation function

#         a_str = str(a).replace(".", "_")
#         self.set_model_name("MLP_Nanofins_2G" + a_str + "Dense1024_U350_H600")
#         self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_2G" + a_str + "Dense1024_U350_H600/")

#         # Define a new architecture
#         def activation_func(x):
#             return gaussian_activation(x, a)

#         self._arch = [
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=activation_func,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=activation_func,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]

# class MLP_Nanofins_GFF2Dense_512_U350_H600_v2(MLP_Nanofins_U350_H600):
#     def __init__(self, emb_dim=256, gauss_scale=10.0, dtype=tf.float64):
#         super().__init__(dtype)

#         d_str = str(int(emb_dim))
#         gscale_str = str(gauss_scale).replace(".", "p")
#         model_name = "MLP_Nanofins_GFF" + d_str + "_" + gscale_str + "Dense512x2_U350_H600_v2"
#         self.set_model_name(model_name)
#         self.set_modelSavePath("trained_MLP_models/" + model_name)

#         # Define a new architecture
#         self._arch = [
#             GFF_Projection_layer_trained(emb_dim, gauss_scale),
#             tf.keras.layers.Dense(
#                 512,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 512,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]


# class MLP_Nanofins_GFF2Dense_1024_U350_H600_v2(MLP_Nanofins_U350_H600):
#     def __init__(self, emb_dim=256, gauss_scale=10.0, dtype=tf.float64):
#         super().__init__(dtype)

#         d_str = str(int(emb_dim))
#         gscale_str = str(gauss_scale).replace(".", "p")
#         model_name = "MLP_Nanofins_GFF" + d_str + "_" + gscale_str + "Dense1024x2_U350_H600_v2"
#         self.set_model_name(model_name)
#         self.set_modelSavePath("trained_MLP_models/" + model_name)

#         # Define a new architecture
#         self._arch = [
#             GFF_Projection_layer_trained(emb_dim, gauss_scale),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation=leakyrelu100,
#                 kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
#             ),
#             tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
#         ]
