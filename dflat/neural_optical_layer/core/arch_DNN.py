import tensorflow as tf
from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600


mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600",
    "MLP_Nanocylinders_Dense128_U180_H600",
    "MLP_Nanofins_Dense1024_U350_H600",
    "MLP_Nanofins_Dense512_U350_H600",
    "MLP_Nanofins_Dense256_U350_H600",
]


def leakyrelu100(x):
    x_pos = (x + tf.abs(x)) / 2
    x_neg = (x - tf.abs(x)) / 2
    return x_pos + 0.01 * x_neg


class MLP_Nanocylinders_Dense256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600/")

        # Define a new architecture
        output_dim = self.get_output_shape()
        arch = [
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
            tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
        self.set_arch(arch)


class MLP_Nanocylinders_Dense128_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype)

        self.set_model_name("MLP_Nanocylinders_Dense128_U180_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanocylinders_Dense128_U180_H600/")

        # Define a new architecture
        output_dim = self.get_output_shape()
        arch = [
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
            tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
        self.set_arch(arch)


class MLP_Nanofins_Dense1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600/")

        # Define a new architecture
        output_dim = self.get_output_shape()
        arch = [
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
            tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
        self.set_arch(arch)


class MLP_Nanofins_Dense512_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense512_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense512_U350_H600/")

        # Define a new architecture
        output_dim = self.get_output_shape()
        arch = [
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
            tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
        self.set_arch(arch)


class MLP_Nanofins_Dense256_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super().__init__(dtype)

        self.set_model_name("MLP_Nanofins_Dense256_U350_H600")
        self.set_modelSavePath("trained_MLP_models/MLP_Nanofins_Dense256_U350_H600/")

        # Define a new architecture
        output_dim = self.get_output_shape()
        arch = [
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
            tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
        self.set_arch(arch)
