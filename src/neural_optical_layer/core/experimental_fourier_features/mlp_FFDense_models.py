import tensorflow as tf
from ..mlp_Dense_models import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600
from .tancik_fourier_feature import FourierFeatureProjection

ffdense_model_names = ["MLP_Nanofins_FFDense_64_U350_H600"]
from ..neural_utilities import leakyrelu100


class MLP_Nanofins_FFDense_64_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(MLP_Nanofins_FFDense_64_U350_H600, self).__init__(**kwargs)
        self.set_model_name("MLP_Nanofins_FFDense_64_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_MLP_models/MLP_Nanofins_FFDense_64_U350_H600/")

        # Define a new architecture
        self._arch = [
            FourierFeatureProjection(12, 6),
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(
                64, activation=leakyrelu100, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
            ),
            tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
        ]

