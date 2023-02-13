import tensorflow as tf
from .arch_Core_class import *
from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600


erbf_model_names = [
    #"ERBF_Nanocylinders_B1024_U180_H600",
    #"ERBF_Nanocylinders_B512_U180_H600",
    #"ERBF_Nanocylinders_B256_U180_H600",
    #"ERBF_Nanofins_B4096_U350_H600",
    #"ERBF_Nanofins_B2048_U350_H600",
    #"ERBF_Nanofins_B1024_U350_H600",
]


## USABLE ERBF Models
class ERBF_Nanocylinders_B1024_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(ERBF_Nanocylinders_B1024_U180_H600, self).__init__(dtype)

        self.set_model_name("ERBF_Nanocylinders_B1024_U180_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanocylinders_B1024_U180_H600/")

        # Define a new architecture
        num_bases = 1024
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]


class ERBF_Nanocylinders_B512_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(ERBF_Nanocylinders_B512_U180_H600, self).__init__(dtype)

        self.set_model_name("ERBF_Nanocylinders_B512_U180_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanocylinders_B512_U180_H600/")

        # Define a new architecture
        num_bases = 512
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]


class ERBF_Nanocylinders_B256_U180_H600(MLP_Nanocylinders_U180_H600):
    def __init__(self, dtype=tf.float64):
        super(ERBF_Nanocylinders_B256_U180_H600, self).__init__(dtype)

        self.set_model_name("ERBF_Nanocylinders_B256_U180_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanocylinders_B256_U180_H600/")

        # Define a new architecture
        num_bases = 256
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(3, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]


##
class ERBF_Nanofins_B4096_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(ERBF_Nanofins_B4096_U350_H600, self).__init__(dtype)

        self.set_model_name("ERBF_Nanofins_B4096_U350_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanofins_B4096_U350_H600/")

        # Define a new architecture
        num_bases = 4096
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]


class ERBF_Nanofins_B2048_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype):
        super(ERBF_Nanofins_B2048_U350_H600, self).__init__(dtype=tf.float64)

        self.set_model_name("ERBF_Nanofins_B2048_U350_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanofins_B2048_U350_H600/")

        # Define a new architecture
        num_bases = 2048
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]


class ERBF_Nanofins_B1024_U350_H600(MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        super(ERBF_Nanofins_B1024_U350_H600, self).__init__(dtype)

        self.set_model_name("ERBF_Nanofins_B1024_U350_H600")
        self.set_modelSavePath("trained_erbf_models/ERBF_Nanofins_B1024_U350_H600/")

        # Define a new architecture
        num_bases = 1024
        rbflayer = EBFLayer(num_bases, initializer=None, betas=1.0)
        outputlayer = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))

        self._arch = [rbflayer, outputlayer]
