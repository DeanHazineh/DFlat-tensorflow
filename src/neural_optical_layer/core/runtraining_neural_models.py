import sys

sys.path.append(".")

import tensorflow as tf
import numpy as np
import pickle

import neural_optical_layer.core.mlp_Dense_models as MLP_models
import neural_optical_layer.core.eRBF_models as eRBF_models
import neural_optical_layer.core.Experimental.mlp_FFDense_models as MLP_ffdense_models
from sklearn.model_selection import train_test_split


def save_test_evaluation_data(model, xtest, ytest):
    modelOut = model.predict(xtest)

    trans_mlp, phase_mlp = model.convert_output_complex(modelOut)
    trans_test, phase_test = model.convert_output_complex(ytest)

    # trans_mlp = trans_mlp.numpy
    phase_mlp = phase_mlp.numpy()
    # trans_test = trans_test.numpy
    phase_test = phase_test.numpy()

    trans_error = trans_test - trans_mlp
    phase_error = phase_test - phase_mlp
    complex_error = trans_mlp * np.exp(1j * phase_mlp) - trans_test * np.exp(1j * phase_test)

    saveTo = model._modelSavePath + "training_testDataError.pickle"
    data = {"trans_error": trans_error, "phase_error": phase_error, "complex_error": complex_error}
    with open(saveTo, "wb") as handle:
        pickle.dump(data, handle)

    return


if __name__ == "__main__":

    ### Define the model to train and associated parameters
    model = MLP_models.MLP_Nanofins_Dense128_U350_H600()
    model.customLoadCheckpoint()

    ### Get training and testing data:
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(
        inputData, outputData, test_size=0.15, random_state=13, shuffle=True
    )

    # Call once then print summary of the model
    model(xtest[0:1, :])
    model.summary()

    train = True
    batch_size = 65536
    device = "GPU:0"
    epochs = 30000
    miniEpoch = 1000  # Number of epochs to save checkpoint after

    # Call Keras model fit api to run training epochs
    if train:
        splitNumberSessions = np.ceil(epochs / miniEpoch).astype("int")
        optimizer = tf.keras.optimizers.Adam(1e-3)
        model.compile(optimizer, loss=tf.keras.losses.mean_squared_error)

        for sessCounter in range(splitNumberSessions):
            with tf.device(device):
                trackhistory = model.fit(
                    xtrain, ytrain, batch_size=batch_size, epochs=miniEpoch, verbose=0, validation_split=0.15
                )

            model.customSaveCheckpoint(trackhistory)

    # After Training, evaluate the performance by histogram of errors on the test set
    save_test_evaluation_data(model, xtest, ytest)
