import tensorflow as tf
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
import numpy as np
import os
import pickle
from keras_flops import get_flops
import matplotlib.pyplot as plt
from pathlib import Path
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import dflat.plot_utilities.graphFunc as graphFunc

### DO NOT ALTER ANYTHING IN THIS FILE IF YOU DON"T KNOW WHAT YOU ARE DOING ELSE IT WILL BREAK THINGS


def get_current_path(folder_name: str):
    resource_path = Path(__file__).parent
    return str(resource_path.joinpath(folder_name)) + "/"


####################################
### Radial Basis function networks
class InitCentersRandom(Initializer):
    """Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X
        super().__init__()

    def __call__(self, shape, dtype=None):
        assert shape[1:] == self.X.shape[1:]  # check dimension

        # np.random.randint returns ints from [low, high) !
        idx = np.random.randint(self.X.shape[0], size=shape[0])

        return self.X[idx, :]


class RBFLayer(tf.keras.layers.Layer):
    """Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):

        self.output_dim = output_dim

        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)

        self.initializer = initializer if initializer else RandomUniform(0.0, 1.0)

        super().__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name="centers", shape=(self.output_dim, input_shape[1]), initializer=self.initializer, trainable=True)
        self.betas = self.add_weight(
            name="betas",
            shape=(1, self.output_dim),
            initializer=self.betas_initializer,
            # initializer='ones',
            trainable=True,
        )

        super().build(input_shape)

    def call(self, x):

        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C - tf.transpose(x))  # matrix of differences

        return tf.exp(-tf.expand_dims(self.betas, 0) * tf.math.reduce_sum(H**2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {"output_dim": self.output_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_centers(self):

        return self.centers


class EBFLayer(tf.keras.layers.Layer):
    """Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # betas is either initializer object or float
        if isinstance(betas, Initializer):
            self.betas_initializer = betas
        else:
            self.betas_initializer = Constant(value=betas)
        self.initializer = initializer if initializer else RandomUniform(0.0, 1.0)

    def build(self, input_shape):
        self.centers = self.add_weight(name="centers", shape=(self.output_dim, input_shape[1]), initializer=self.initializer, trainable=True)
        self.betas = self.add_weight(
            name="betas",
            shape=(input_shape[1], self.output_dim),
            initializer=self.betas_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        C = tf.expand_dims(self.centers, -1)  # inserts a dimension of 1
        H = tf.transpose(C - tf.transpose(x))  # matrix of differences

        return tf.exp(-tf.math.reduce_sum(H**2 / tf.expand_dims(self.betas, 0) ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {"output_dim": self.output_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_centers(self):

        return self.centers


class Cov_EBFLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, initializer=None, betas=None, betas_init_value=10.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.betas_initializer = initializer if initializer else Constant(value=betas_init_value)  # RandomUniform(0.0, 1.0)
        self.initializer = initializer if initializer else RandomUniform(0.0, 1.0)

    def build(self, input_shape):
        self.centers = self.add_weight(name="centers", shape=(self.output_dim, 1, input_shape[1]), initializer=self.initializer, trainable=True)
        self.betas = self.add_weight(name="betas", shape=(self.output_dim, input_shape[1], input_shape[1]), initializer=self.betas_initializer, trainable=True)
        super().build(input_shape)

    def call(self, x):
        precision = tf.expand_dims(tf.linalg.matmul(self.betas, tf.transpose(self.betas, [0, 2, 1])), 1)
        H = tf.expand_dims(self.centers - tf.expand_dims(x, 0), -1)

        out = tf.squeeze(tf.linalg.matmul(tf.linalg.matmul(H, precision, transpose_a=True), H), axis=(2, 3))
        out = tf.transpose(tf.exp(-out), [1, 0])

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {"output_dim": self.output_dim}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_centers(self):
        return self.centers


################################
### Multivariate Polynomial Models
class Multivariate_Polynomial_Object:
    def __init__(self, **kwargs):
        self.__input_features = 0
        self.__output_features = 0
        self.__poly_degree = 0
        self.__holdPolyObjects = []
        self.__polynomial_features = []

        return

    def build_poly(self):
        # Create Poly Coefficient Objects
        for i in range(self.__output_features):
            polyModel = LinearRegression()
            self.__holdPolyObjects.append(polyModel)

        # Create Poly Object for usage
        self._interactionOnly = False
        self.__polynomial_features = PolynomialFeatures(self.__poly_degree, interaction_only=self._interactionOnly)

        return

    def _set_input_features(self, input_features):
        self.__input_features = input_features
        return

    def _set_output_features(self, output_features):
        self.__output_features = output_features

        return

    def _set_poly_degree(self, poly_degree):
        self.__poly_degree = poly_degree

    def save_test_evaluation_data(self, xtest, ytest, savestring):
        poly_pred_train = self.model_predict(xtest)
        pred_trans, pred_phase = self.convert_output_complex(poly_pred_train)
        true_trans, true_phase = self.convert_output_complex(ytest)

        trans_error = pred_trans - true_trans
        phase_error = pred_phase - true_phase
        complex_error = np.abs(pred_trans * np.exp(1j * pred_phase.numpy()) - true_trans * np.exp(1j * true_phase.numpy()))

        # get relative errors
        rel_trans = trans_error / true_trans
        rel_phase = phase_error / true_phase
        rel_complex = complex_error / np.abs(true_trans * np.exp(1j * true_phase.numpy()))

        # Estimate FLOPs
        est_FLOPs = self.profile_FLOPs()

        saveTo = self._modelSavePath + savestring
        data = {
            "trans_error": trans_error,
            "rel_trans": rel_trans,
            "phase_error": phase_error,
            "rel_phase": rel_phase,
            "complex_error": complex_error,
            "rel_complex": rel_complex,
            "est_FLOPs": est_FLOPs,
        }
        with open(saveTo, "wb") as handle:
            pickle.dump(data, handle)

        return complex_error

    def fit_polyCoeff(self):
        start_time = time.time()
        # Get Data
        inputData, outputData = self.returnLibraryAsTrainingData()
        xtrain, xtest, ytrain, ytest = train_test_split(inputData, outputData, test_size=0.15, random_state=13, shuffle=True)

        # Fit to training data
        polyFeatures_train = self.__polynomial_features.fit_transform(xtrain)
        for i in range(self.__output_features):
            self.__holdPolyObjects[i].fit(polyFeatures_train, ytrain[:, i])
        end_time = time.time()

        # Save model fit
        print("Saving Model Fit")
        self.customSaveCheckpoint()

        # Save error data
        complex_error_test = self.save_test_evaluation_data(xtest, ytest, "training_testDataError.pickle")
        complex_error_train = self.save_test_evaluation_data(xtrain, ytrain, "training_trainDataError.pickle")

        print("time | MAE: ", end_time - start_time, np.mean(complex_error_test), np.mean(complex_error_train))

        return

    def model_predict(self, model_input):
        polyFeatures_input = self.__polynomial_features.fit_transform(model_input)

        output = []
        for i in range(self.__output_features):
            output.append(self.__holdPolyObjects[i].predict(polyFeatures_input))
        output = np.transpose(np.vstack(output), [1, 0])

        return output

    def predict(self, model_input):
        return self.model_predict(model_input)

    def __call__(self, model_input):
        return self.model_predict(model_input)

    def customLoadCheckpoint(self):

        print("Checking for model checkpoint at: " + self._modelSavePath)
        pkl_filename = self._modelSavePath + "model.pickle"

        if os.path.exists(pkl_filename):
            with open(pkl_filename, "rb") as file:
                data = pickle.load(file)

                self.__input_features = data["input_features"]
                self.__output_features = data["output_features"]
                self.__poly_degree = data["poly_degree"]
                self.__holdPolyObjects = data["holdPolyObject"]

            print("\n Model Checkpoint Loaded \n")

        return

    def customSaveCheckpoint(self):
        # save weights to checkpoint file
        pkl_filename = self._modelSavePath + "model.pickle"

        data = {
            "input_features": self.__input_features,
            "output_features": self.__output_features,
            "poly_degree": self.__poly_degree,
            "holdPolyObject": self.__holdPolyObjects,
        }
        with open(pkl_filename, "wb") as file:
            pickle.dump(data, file)

        print("\n Model Saved \n")

        return

    def profile_FLOPs(self):
        # FLOPs of matrix multiplication
        # (1 x num_polyCoeffs_per_output) _matmul_ (num_polyCoeffs_per_output x outsize)
        num_model_input_features = self.__input_features
        num_polyCoeffs_per_output = math.comb(num_model_input_features + self.__poly_degree, self.__poly_degree)

        estFLOPs = 1 * self.__output_features * (2 * num_polyCoeffs_per_output - 1)
        print(
            "Degree; Params; Est FLOPs | ",
            self.__poly_degree,
            num_polyCoeffs_per_output * self.__output_features,
            estFLOPs,
        )

        return estFLOPs


#################################
#### MLP Models
class GFF_Projection_layer(tf.keras.layers.Layer):
    def __init__(self, gaussian_projection: int, gaussian_scale: float = 1.0, **kwargs):
        """
        Fourier Feature Projection layer from the paper:
        [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
        Add this layer immediately after the input layer.
        Args:
            gaussian_projection: Projection dimension for the gaussian kernel in fourier feature
                projection layer. Can be negative or positive integer.
                If <=0, uses identity matrix (basic projection) without gaussian kernel.
                If >=1, uses gaussian projection matrix of specified dim.
            gaussian_scale: Scale of the gaussian kernel in fourier feature projection layer.
                Note: If the scale is too small, convergence will slow down and obtain poor results.
                If the scale is too large (>50), convergence will be fast but results will be grainy.
        """
        super().__init__(**kwargs)

        if "dtype" in kwargs:
            self._kernel_dtype = kwargs["dtype"]
        else:
            self._kernel_dtype = None

        self.gauss_proj = int(gaussian_projection)
        self.gauss_scale = float(gaussian_scale)

    def build(self, input_shape):
        # assume channel dim is always at last location
        input_dim = input_shape[-1]

        if self.gauss_proj <= 0:
            self.proj_kernel = tf.keras.layers.Dense(input_dim, use_bias=True, trainable=True, kernel_initializer="identity", dtype=self._kernel_dtype)
        else:
            initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=self.gauss_scale)
            self.proj_kernel = tf.keras.layers.Dense(
                self.gauss_proj,
                use_bias=True,
                trainable=True,
                kernel_initializer=initializer,
                dtype=self._kernel_dtype,
            )

        self.built = True

    def call(self, inputs):
        x_proj = self.proj_kernel(2.0 * np.pi * inputs)

        x_proj_sin = tf.sin(x_proj)
        x_proj_cos = tf.cos(x_proj)

        output = tf.concat([x_proj_sin, x_proj_cos], axis=-1)
        return output

    def get_config(self):
        config = {"gaussian_projection": self.gauss_proj, "gaussian_scale": self.gauss_scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MLP_Object(tf.keras.Model):
    def __init__(self):
        super(MLP_Object, self).__init__()

        # Define class variables
        self._modelSavePath = ""
        self.trainingLoss = []
        self.trainingValLoss = []
        self.__model_name = ""
        self.__accepts_wavelength = True
        self._dtype = tf.float64
        self.__input_shape_tuple = 0
        self.__output_pol_state = 1

        # parameter limits wrapped into a list for generalized model usage
        self.__preprocessDataBounds = []
        self.__dataBoundsLabel = []
        self._arch = []

    def set_model_dtype(self, dtype):
        self._dtype = dtype
        return

    def get_model_dtype(self):
        return self._dtype

    def set_modelSavePath(self, modelSavePath):
        self._modelSavePath = get_current_path(modelSavePath)

        if not os.path.exists(self._modelSavePath):
            os.makedirs(modelSavePath, exist_ok=True)
            os.makedirs(modelSavePath + "/trainingOutput/", exist_ok=True)

        # Make folders for images too
        if not os.path.exists(self._modelSavePath + "trainingOutput/png_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/png_images/", exist_ok=True)

        if not os.path.exists(self._modelSavePath + "trainingOutput/pdf_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/pdf_images/", exist_ok=True)

        return

    def set_preprocessDataBounds(self, preprocessDataBounds, boundLabels):
        self.__preprocessDataBounds = preprocessDataBounds
        self.__dataBoundsLabel = boundLabels
        return

    def get_preprocessDataBounds(self):
        return self.__preprocessDataBounds

    def set_model_name(self, name):
        self.__model_name = name
        return

    def get_model_name(self):
        return self.__model_name

    def set_wavelengthFlag(self, boolFlag):
        self.__accepts_wavelength = boolFlag
        return

    def get_wavelengthFlag(self):
        return self.__accepts_wavelength

    def set_input_shape(self, input_shape):
        self.__input_shape_tuple = input_shape
        return

    def get_input_shape(self):
        return self.__input_shape_tuple

    def set_output_pol_state(self, output_stack_num):
        self.__output_pol_state = output_stack_num
        return

    def get_output_pol_state(self):
        return self.__output_pol_state

    def call(self, y):
        for layer in self._arch:
            y = layer(y)
        y = tf.cast(y, dtype=self._dtype)
        return y

    def normalizeInput(self, paramList):
        # take in a list of parameters and normalize
        # based on the class pre-defined parameter limits
        # Normalized input parameters to [0,1]
        # Ensures stability with NN initialization and compatability with constrained optimization
        outParams = []

        for counter, thisParam in enumerate(paramList):
            parameterBounds = self.__preprocessDataBounds[counter]
            outParams.append((thisParam - parameterBounds[0]) / (parameterBounds[1] - parameterBounds[0]))

        return outParams

    def normalizeWavelength(self, wavelength_m):
        with tf.debugging.assert_equal(
            self.__dataBoundsLabel[-1],
            "wavelength_m",
            message="wavelength should have been the last listed parameter",
            name="preprocessDataBound format assertion",
        ):
            wavelength_preprocessBounds = self.__preprocessDataBounds[-1]
            wavelength_mlp = (wavelength_m - wavelength_preprocessBounds[0]) / (wavelength_preprocessBounds[1] - wavelength_preprocessBounds[0])

        return wavelength_mlp

    def convert_vectorParam_toMLPInput(self, paramList_asvector):
        ### Sometimes desire mlp output in meshgrid form with vector axis labels.
        # this is just a convenient wrapper to call mlp output on a grid without having
        # to call meshgrid in the main script.
        paramlist_asgrid = np.meshgrid(*paramList_asvector)
        outParams = self.normalizeInput(paramlist_asgrid)

        return np.stack([param.flatten() for param in outParams], -1)

    def customLoadCheckpoint(self):
        ## Custom models require their own functions to handle loads and saves
        # If a checkpoint file exists then load the checkpoint weights to architecture
        print("Checking for model checkpoint at: " + self._modelSavePath)
        if os.path.exists(self._modelSavePath + "checkpoint"):
            self.load_weights(self._modelSavePath).expect_partial()
            print("\n Model Checkpoint Loaded \n")
        else:
            print("\n no model checkpoint found at\n", self._modelSavePath + "checkpoint")

        # Load the previous training loss vector if it exists
        if os.path.exists(self._modelSavePath + "trainingHistory.pickle"):
            with open(self._modelSavePath + "trainingHistory.pickle", "rb") as handle:
                trackHistory = pickle.load(handle)
                self.trainingLoss = trackHistory["trainingLoss"]
                self.trainingValLoss = trackHistory["trainingValLoss"]

        return

    def customSaveCheckpoint(self, trackHistoryObject=[]):
        # save weights to checkpoint file
        self.save_weights(self._modelSavePath)
        print("\n Model Saved \n")

        # if trackHistory keras object is passed then manually update by concatenating
        # loss vector to current model loss vector and saving current state to pickle
        # Also save a plot displaying the loss state during training for convenience!
        if trackHistoryObject:
            self.trainingLoss = np.concatenate((self.trainingLoss, trackHistoryObject.history["loss"]))
            self.trainingValLoss = np.concatenate((self.trainingValLoss, trackHistoryObject.history["val_loss"]))

            data = {
                "trainingLoss": self.trainingLoss,
                "trainingValLoss": self.trainingValLoss,
            }
            pickle.dump(data, open(self._modelSavePath + "trainingHistory.pickle", "wb"))

            fig = plt.figure(figsize=(20, 10))
            ax = graphFunc.addAxis(fig, 1, 2)
            ax[0].plot(self.trainingLoss, "b-.", label="training loss")
            ax[0].plot(self.trainingValLoss, "r-.", label="validation loss")
            graphFunc.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addLegend=True)

            ax[1].plot(np.log10(self.trainingLoss), "b-.")
            ax[1].plot(np.log10(self.trainingValLoss), "r-.")
            graphFunc.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")

            plt.savefig(self._modelSavePath + "/trainingOutput/png_images/trainingLog_traininghistory.png")
            plt.savefig(self._modelSavePath + "/trainingOutput/pdf_images/trainingLog_traininghistory.pdf")
            plt.close()

        return

    def profile_FLOPs(self):
        # To use keras-flops, we need the architecture defined via keras sequential
        layers = []
        layers.append(tf.keras.Input(shape=self.__input_shape_tuple))

        for layer in self._arch:
            layers.append(layer)

        # Use keras Flops as one metric
        tempModel = tf.keras.Sequential(layers)
        estFLOPs = get_flops(tempModel, batch_size=1)
        print("FLOPs Analysis 1: Keras_Flops: ", estFLOPs)
        print("\n ======================================= \n ")

        ## Use second function
        # print("The v2 FLOPs is:{}".format(get_flops_alternate(tempModel)), flush=True)

        return estFLOPs
