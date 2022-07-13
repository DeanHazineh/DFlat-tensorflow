import tensorflow as tf
from .vidnerova_rbfLayer import *
from .mlp_Dense_models import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600


erbf_model_names = ["ERBF_Nanocylinders_B250_U180_H600", "ERBF_Nanofins_B1024_U350_H600"]


## USABLE ERBF Models
class ERBF_Nanocylinders_B250_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, **kwargs):
        super(ERBF_Nanocylinders_B250_U180_H600, self).__init__(**kwargs)

        self.set_model_name("ERBF_Nanocylinders_B250_U180_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_erbf_models/ERBF_Nanocylinders_B250_U180_H600/")

        # Define a new architecture
        num_bases = 250
        num_pred = 3
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(num_pred, use_bias=False)

        self._arch = [rbflayer, outputlayer]


class ERBF_Nanofins_B1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, **kwargs):
        super(ERBF_Nanofins_B1024_U350_H600, self).__init__(**kwargs)

        self.set_model_name("ERBF_Nanofins_B1024_U350_H600")
        self.set_modelSavePath("neural_optical_layer/core/trained_erbf_models/ERBF_Nanofins_B1024_U350_H600/")

        # Define a new architecture
        num_bases = 1024
        num_pred = 6
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(num_pred, use_bias=False)

        self._arch = [rbflayer, outputlayer]
