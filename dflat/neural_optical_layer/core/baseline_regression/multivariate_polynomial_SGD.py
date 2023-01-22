import math
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import time

import dflat.plot_utilities.graphFunc as graphFunc


class multi_poly_regression(tf.keras.Model):
    def __init__(self, model_name, polyDegree, num_features, num_outputs):
        super(multi_poly_regression, self).__init__()
        self.set_modelSavePath("interpolant_optical_layer/core/polynomial_models/" + model_name + "/")
        self.trainingLoss = []
        self.trainingValLoss = []

        # Initialize shape dimensions
        self.polyDegree = polyDegree
        num_polyFeatures = math.comb(num_features + polyDegree, polyDegree)
        print("Number Coefficients Poly Model: ", num_polyFeatures * num_outputs)

        # Use the poly Features class to transform input data
        self.poly = PolynomialFeatures(polyDegree, interaction_only=False)

        # Initialize the poly Coeffs matrix
        init_coeffs = tf.random.uniform(
            shape=(num_polyFeatures, num_outputs), minval=0, maxval=1.0, dtype=tf.float32, seed=13
        )
        self.polyCoeffs = tf.Variable(init_coeffs, trainable=True)

    def call(self, inputDat):
        input_polyFeatures = tf.convert_to_tensor(self.poly.fit_transform(inputDat), tf.float32)
        return tf.linalg.matmul(input_polyFeatures, self.polyCoeffs)

    def set_modelSavePath(self, modelSavePath):
        self._modelSavePath = modelSavePath

        if not os.path.exists(self._modelSavePath):
            os.makedirs(modelSavePath)
            os.makedirs(modelSavePath + "/trainingOutput/")

        # Make folders for images too
        if not os.path.exists(self._modelSavePath + "trainingOutput/png_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/png_images/")

        if not os.path.exists(self._modelSavePath + "trainingOutput/pdf_images/"):
            os.makedirs(self._modelSavePath + "trainingOutput/pdf_images/")

        return

    def customLoadCheckpoint(self):
        ## Custom models require their own functions to handle loads and saves

        # If a checkpoint file exists then load the checkpoint weights to architecture
        print("Checking for model checkpoint at: " + self._modelSavePath)
        if os.path.exists(self._modelSavePath + "checkpoint"):
            self.load_weights(self._modelSavePath).expect_partial()
            print("\n Model Checkpoint Loaded \n")

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
            self.trainingLoss = np.concatenate((self.trainingLoss, trackHistoryObject["loss"]))
            self.trainingValLoss = np.concatenate((self.trainingValLoss, trackHistoryObject["val_loss"]))

            data = {
                "trainingLoss": self.trainingLoss,
                "trainingValLoss": self.trainingValLoss,
            }
            pickle.dump(data, open(self._modelSavePath + "trainingHistory.pickle", "wb"))

            fig = plt.figure(figsize=(20, 10))
            ax = graphFunc.addAxis(fig, 1, 2)
            ax[0].plot(self.trainingLoss, "b-.", label="training loss")
            ax[0].plot(self.trainingValLoss, "r-.", label="validation loss")
            graphFunc.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss", addlegend=True)

            ax[1].plot(np.log10(self.trainingLoss), "b-.")
            ax[1].plot(np.log10(self.trainingValLoss), "r-.")
            graphFunc.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Traning Log(Loss)")

            plt.savefig(self._modelSavePath + "/trainingOutput/png_images/trainingLog_traininghistory.png")
            plt.savefig(self._modelSavePath + "/trainingOutput/pdf_images/trainingLog_traininghistory.pdf")
            plt.close()

        return


def mse_loss_fn(regression, input_dat, output_dat):
    return tf.math.reduce_mean(tf.math.square(regression(input_dat) - output_dat))


def train_step(regression, input_batch, output_batch, optimizer):
    with tf.GradientTape() as tape:
        current_loss = mse_loss_fn(regression, input_batch, output_batch)

    gradients = tape.gradient(current_loss, regression.trainable_variables)
    optimizer.apply_gradients(zip(gradients, regression.trainable_variables))

    return current_loss.numpy()


def train_regression(
    regression,
    input_data,
    output_data,
    input_val_data,
    output_val_data,
    batch_size,
    num_epochs,
    optimizer,
    term_grad_loss,
    verbose=False,
):
    num_batches = input_data.shape[0] // batch_size
    leftover_num = input_data.shape[0] - (num_batches * batch_size)
    partial_batch_start = num_batches * batch_size

    epoch_loss = []
    val_loss = []
    for epoch in range(num_epochs):

        # Loop over minibatches of data
        start = time.time()
        batch_loss = []
        for minibatch in range(num_batches):
            current_loss = train_step(
                regression,
                input_data[(minibatch * batch_size) : (minibatch + 1) * batch_size, :],
                output_data[(minibatch * batch_size) : (minibatch + 1) * batch_size, :],
                optimizer,
            )
            batch_loss.append(current_loss)
        if leftover_num != 0:  # final partial batch
            current_loss = train_step(
                regression, input_data[partial_batch_start:, :], output_data[partial_batch_start:, :], optimizer,
            )
            batch_loss.append(current_loss)

        # Save the epoch loss
        current_loss = np.mean(batch_loss)
        epoch_loss.append(current_loss)

        # Compute the validation loss
        current_val_loss = mse_loss_fn(regression, input_val_data, output_val_data).numpy()
        val_loss.append(current_val_loss)

        end = time.time()
        if verbose:
            print("Training Log | (epoch, time, loss, val_loss)", epoch, end - start, current_loss, current_val_loss)

        # Terminate based on moving window average of gradients
        if epoch > 10:
            grad_loss_mean = np.mean(np.abs(np.array(val_loss)[-6:-1] - np.array(val_loss)[-5:]))
            if grad_loss_mean < term_grad_loss:
                break

    # Post training, save the model
    trackHistoryObject = {"loss": epoch_loss, "val_loss": val_loss}
    regression.customSaveCheckpoint(trackHistoryObject)

    return trackHistoryObject


def single_poly_SGD_train(
    model_name, modelClass, polyDegree, lr=1, term_grad_loss=1e-5, use_ckpt=False, max_epochs=1000, verbose=False
):
    ### Load the input and output data, using same seed and split as the MLP
    # The MLP had a validation set which pulled out 15% of training data
    model = modelClass()
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(
        inputData, outputData, test_size=0.15, random_state=13, shuffle=True
    )
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.15, random_state=13, shuffle=True)

    # Initialize the poly regression class
    num_features = xtrain.shape[1]
    num_outputs = ytrain.shape[1]
    poly_regression = multi_poly_regression(model_name, polyDegree, num_features, num_outputs)
    if use_ckpt:
        poly_regression.customLoadCheckpoint()

    optimizer = tf.keras.optimizers.Adam(lr)
    trackHistory = train_regression(
        poly_regression, xtrain, ytrain, xval, yval, 60000, max_epochs, optimizer, term_grad_loss, verbose=True,
    )

    return trackHistory
