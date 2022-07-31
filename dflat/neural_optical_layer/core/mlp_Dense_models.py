import tensorflow as tf
from .neural_utilities import leakyrelu100
from .mlp_parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600

mlp_model_names = [
    "MLP_Nanocylinders_Dense256_U180_H600",
    "MLP_Nanocylinders_Dense128_U180_H600",
    "MLP_Nanocylinders_Dense64_U180_H600",
    # "MLP_Nanocylinders_Dense32_U180_H600",
    "MLP_Nanofins_Dense1024_U350_H600",
    # "MLP_Nanofins_Dense512_U350_H600",
    "MLP_Nanofins_Dense256_U350_H600",
    "MLP_Nanofins_Dense128_U350_H600",
    "MLP_Nanofins_Dense64_U350_H600",
    "MLP_Nanofins_Dense32_U350_H600",
]


## USABLE MLP MODELS
class MLP_Nanocylinders_Dense256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense256_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense256_U180_H600")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense256_U180_H600/"
        )

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
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense128_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense128_U180_H600")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense128_U180_H600/"
        )

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
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense64_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense64_U180_H600")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense64_U180_H600/"
        )

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


class MLP_Nanocylinders_Dense32_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanocylinders_Dense32_U180_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanocylinders_Dense32_U180_H600")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanocylinders_Dense32_U180_H600/"
        )

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                32,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                32,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


##
class MLP_Nanofins_Dense1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense1024_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense1024_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense1024_U350_H600/")

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
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense512_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense512_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense512_U350_H600/")

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
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense256_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense256_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense256_U350_H600/")

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


class MLP_Nanofins_Dense128_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense128_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense128_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense128_U350_H600/")

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
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]


class MLP_Nanofins_Dense64_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense64_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense64_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense64_U350_H600/")

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


class MLP_Nanofins_Dense32_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_Dense32_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_Dense32_U350_H600")
        self.set_modelSavePath("dflat/neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_Dense32_U350_H600/")

        # Define a new architecture
        self._arch = [
            tf.keras.layers.Dense(
                32,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                32,
                activation=leakyrelu100,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]
