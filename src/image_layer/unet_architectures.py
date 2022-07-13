import tensorflow as tf
from tensorflow.keras import activations


class Unet_6Depth_2Block_16Filter:
    def __init__(self, input_im_shape, init_kernel_size, kernel_size=4):
        self.input_im_shape = input_im_shape
        self.init_kernel_size = init_kernel_size
        self.kernel_size = kernel_size

        return

    def build_model(self):
        activation = activations.relu

        inputs = tf.keras.layers.Input(shape=(self.input_im_shape[0], self.input_im_shape[1], self.input_im_shape[2]))

        c0 = tf.keras.layers.Conv2D(16, kernel_size=self.init_kernel_size, padding="same", activation=activation)(
            inputs
        )
        c1 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c0)
        c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        c3 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c2)
        c4 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c3)
        c5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

        c6 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c5)
        c7 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c6)
        c8 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c7)

        c9 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c8)
        c10 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c9)
        c11 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c10)

        c12 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c11)
        c13 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c12)
        c14 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c13)

        c15 = tf.keras.layers.Conv2D(512, kernel_size=self.kernel_size, padding="same", activation=activation)(c14)
        c16 = tf.keras.layers.Conv2D(512, kernel_size=self.kernel_size, padding="same", activation=activation)(c15)
        c17 = tf.keras.layers.Conv2DTranspose(
            256, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c16)

        c18 = tf.keras.layers.concatenate([c17, c13], axis=-1)
        c19 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c18)
        c20 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c19)
        c21 = tf.keras.layers.Conv2DTranspose(
            128, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c20)

        c22 = tf.keras.layers.concatenate([c21, c10], axis=-1)
        c23 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c22)
        c24 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c23)
        c25 = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c24)

        c26 = tf.keras.layers.concatenate([c25, c7], axis=-1)
        c27 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c26)
        c28 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c27)
        c29 = tf.keras.layers.Conv2DTranspose(
            32, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c28)

        c30 = tf.keras.layers.concatenate([c29, c4], axis=-1)
        c31 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c30)
        c32 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c31)
        c33 = tf.keras.layers.Conv2DTranspose(
            16, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c32)

        c34 = tf.keras.layers.concatenate([c33, c1], axis=-1)
        c35 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c34)
        c36 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c35)
        c37 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation=activation)(c36)

        model = tf.keras.Model(inputs=inputs, outputs=c37, name="Unet_6Depth_2Block_16Filter")

        return model


class Unet_5Depth_2Block_16Filter:
    def __init__(self, input_im_shape, init_kernel_size, kernel_size=4):
        self.input_im_shape = input_im_shape
        self.init_kernel_size = init_kernel_size
        self.kernel_size = kernel_size

        return

    def build_model(self):
        activation = activations.relu

        inputs = tf.keras.layers.Input(shape=(self.input_im_shape[0], self.input_im_shape[1], self.input_im_shape[2]))

        c0 = tf.keras.layers.Conv2D(16, kernel_size=self.init_kernel_size, padding="same", activation=activation)(
            inputs
        )
        c1 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c0)
        c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        c3 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c2)
        c4 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c3)
        c5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

        c6 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c5)
        c7 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c6)
        c8 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c7)

        c9 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c8)
        c10 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c9)
        c11 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c10)

        c12 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c11)
        c13 = tf.keras.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation=activation)(c12)
        c14 = tf.keras.layers.Conv2DTranspose(
            128, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c13)

        c15 = tf.keras.layers.concatenate([c14, c10], axis=-1)
        c16 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c15)
        c17 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c16)
        c18 = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c17)

        c19 = tf.keras.layers.concatenate([c18, c7], axis=-1)
        c20 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c19)
        c21 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c20)
        c22 = tf.keras.layers.Conv2DTranspose(
            32, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c21)

        c23 = tf.keras.layers.concatenate([c22, c4], axis=-1)
        c24 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c23)
        c25 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c24)
        c26 = tf.keras.layers.Conv2DTranspose(
            16, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c25)

        c27 = tf.keras.layers.concatenate([c26, c1], axis=-1)
        c28 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c27)
        c29 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c28)
        c30 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation=activation)(c29)

        model = tf.keras.Model(inputs=inputs, outputs=c30, name="Unet_Layer_5_2Block_16Filter")

        return model


class Unet_4Depth_2Block_16Filter:
    def __init__(self, input_im_shape, init_kernel_size, kernel_size=4):
        self.input_im_shape = input_im_shape
        self.init_kernel_size = init_kernel_size
        self.kernel_size = kernel_size

        return

    def build_model(self):
        activation = "tanh"
        inputs = tf.keras.layers.Input(shape=(self.input_im_shape[0], self.input_im_shape[1], self.input_im_shape[2]))

        c0 = tf.keras.layers.Conv2D(16, kernel_size=self.init_kernel_size, padding="same", activation=activation)(
            inputs
        )
        c1 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c0)
        c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        c3 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c2)
        c4 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c3)
        c5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

        c6 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c5)
        c7 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c6)
        c8 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c7)

        c9 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c8)
        c10 = tf.keras.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation=activation)(c9)
        c11 = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c10)

        c12 = tf.keras.layers.concatenate([c11, c7], axis=-1)
        c13 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c12)
        c14 = tf.keras.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation=activation)(c13)
        c15 = tf.keras.layers.Conv2DTranspose(
            32, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c14)

        c16 = tf.keras.layers.concatenate([c15, c4], axis=-1)
        c17 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c16)
        c18 = tf.keras.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation=activation)(c17)
        c19 = tf.keras.layers.Conv2DTranspose(
            16, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c18)

        c20 = tf.keras.layers.concatenate([c19, c1], axis=-1)
        c21 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c20)
        c22 = tf.keras.layers.Conv2D(16, kernel_size=self.kernel_size, padding="same", activation=activation)(c21)
        c23 = tf.keras.layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation=activation)(c22)

        model = tf.keras.Model(inputs=inputs, outputs=c23, name="Unet_Layer_4_2Block_16Filter")

        return model


class Unet_Layer_3_2Block_16Filter(tf.keras.layers.Layer):
    def __init__(self, input_im_shape):
        super(Unet_Layer_3_2Block_16Filter, self).__init__()
        self.input_im_shape = input_im_shape

    def build_model(self):
        activation = "tanh"
        inputs = tf.keras.layers.Input(shape=(self.input_im_shape[0], self.input_im_shape[1], self.input_im_shape[2]))
        c0 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(inputs)
        c1 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c0)
        c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        c3 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c2)
        c4 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c3)
        c5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

        c6 = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=activation)(c5)
        c7 = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=activation)(c6)
        c8 = tf.keras.layers.Conv2DTranspose(32, kernel_size=2, strides=(2, 2), padding="same", activation=activation)(
            c7
        )

        c9 = tf.keras.layers.concatenate([c8, c4], axis=-1)
        c10 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c9)
        c11 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c10)
        c12 = tf.keras.layers.Conv2DTranspose(
            16, kernel_size=2, strides=(2, 2), padding="same", activation=activation
        )(c11)

        c13 = tf.keras.layers.concatenate([c12, c1], axis=-1)
        c14 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c13)
        c15 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c14)

        c16 = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=activation)(c15)

        model = tf.keras.Model(inputs=inputs, outputs=c16, name="Unet_Layer_3_2Block_16Filter")

        return model


class Unet_Layer_3_2Block_8Filter(tf.keras.layers.Layer):
    def __init__(self, input_im_shape):
        super(Unet_Layer_3_2Block_8Filter, self).__init__()
        self.input_im_shape = input_im_shape

    def build_model(self):
        activation = "tanh"
        inputs = tf.keras.layers.Input(shape=(self.input_im_shape[0], self.input_im_shape[1], self.input_im_shape[2]))
        c0 = tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", activation=activation)(inputs)
        c1 = tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", activation=activation)(c0)
        c2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c1)

        c3 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c2)
        c4 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c3)
        c5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(c4)

        c6 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c5)
        c7 = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=activation)(c6)
        c8 = tf.keras.layers.Conv2DTranspose(16, kernel_size=2, strides=(2, 2), padding="same", activation=activation)(
            c7
        )

        c9 = tf.keras.layers.concatenate([c8, c4], axis=-1)
        c10 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c9)
        c11 = tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", activation=activation)(c10)
        c12 = tf.keras.layers.Conv2DTranspose(8, kernel_size=2, strides=(2, 2), padding="same", activation=activation)(
            c11
        )

        c13 = tf.keras.layers.concatenate([c12, c1], axis=-1)
        c14 = tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", activation=activation)(c13)
        c15 = tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", activation=activation)(c14)

        c16 = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=activation)(c15)

        model = tf.keras.Model(inputs=inputs, outputs=c16, name="Unet_Layer_3_2Block_16Filter")

        return model

