import tensorflow as tf
import numpy as np
import pickle

import dflat.neural_optical_layer.core.mlp_Dense_models as MLP_models
import dflat.neural_optical_layer.core.eRBF_models as eRBF_models
from sklearn.model_selection import train_test_split


def save_test_evaluation_data(model, xtest, ytest, savestring):
    # Get model errors on the test set
    modelOut = model.predict(xtest, verbose=0)
    trans_mlp, phase_mlp = model.convert_output_complex(modelOut)
    trans_test, phase_test = model.convert_output_complex(ytest)
    phase_mlp = phase_mlp.numpy()
    phase_test = phase_test.numpy()

    # Compute errors
    trans_error = trans_test - trans_mlp
    phase_error = phase_test - phase_mlp
    complex_error = np.abs(trans_mlp * np.exp(1j * phase_mlp) - trans_test * np.exp(1j * phase_test))

    # compute relative errors
    rel_trans = trans_error / trans_test
    rel_phase = phase_error / phase_test
    rel_complex = complex_error / np.abs(trans_test * np.exp(1j * phase_test))

    # Est FLOPs per evaluation
    est_FLOPs = model.profile_FLOPs()

    saveTo = model._modelSavePath + savestring + ".pickle"
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


def run_training_neural_model(
    model_caller, epochs, miniEpoch=1000, batch_size=65536, lr=1e-4, verbose=False, train=True
):

    ### Define the model to train and associated parameters
    model = model_caller()
    model.customLoadCheckpoint()

    ### Get training and testing data:
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(
        inputData, outputData, test_size=0.15, random_state=13, shuffle=True
    )

    # Call once then print summary
    model(xtrain[0:1, :])
    model.summary()

    train = train
    device = "GPU:0"
    val_loss_window = 250

    # Call Keras model fit api to run training epochs
    if train:
        splitNumberSessions = np.ceil(epochs / miniEpoch).astype("int")
        optimizer = tf.keras.optimizers.Adam(lr)
        model.compile(optimizer, loss=tf.keras.losses.mean_squared_error)

        val_loss_vec = []
        for sessCounter in range(splitNumberSessions):
            with tf.device(device):
                trackhistory = model.fit(
                    xtrain, ytrain, batch_size=batch_size, epochs=miniEpoch, verbose=verbose, validation_split=0.15
                )
            model.customSaveCheckpoint(trackhistory)

            # Allow for gradient based termination
            # val_loss_vec = val_loss_vec + trackhistory.history["val_loss"]
            # if (sessCounter + 1) * miniEpoch > val_loss_window:
            #     val_loss_set = val_loss_vec[-val_loss_window:]
            #     avg_grad_window = np.mean(np.array(val_loss_set[1:]) - np.array(val_loss_set[0:-1]))
            #     avg_grad_window = np.sign(avg_grad_window) * np.sqrt(np.abs(avg_grad_window))
            #     print("Sess | grad sqrt(mse val): ", sessCounter, avg_grad_window)
            #     if avg_grad_window < 0 and avg_grad_window > -1e-5:
            #         print("Gradient based early termination")
            #         break

    # After Training, evaluate the performance by histogram of errors on the test set
    test_complex_error = save_test_evaluation_data(model, xtest, ytest, "training_testDataError")
    save_test_evaluation_data(model, xtrain, ytrain, "training_trainDataError")

    print("MAE Test Set: ", np.mean(np.abs(test_complex_error)))
    return


##
def train_caller(train=True, verb=False):
    # run_training_neural_model(
    #     model_caller=MLP_models.MLP_Nanofins_Dense256_U350_H600,
    #     epochs=10000,
    #     miniEpoch=200,
    #     lr=1e-4,
    #     train=train,
    #     verbose=verb,
    # )

    run_training_neural_model(
        model_caller=MLP_models.MLP_Nanofins_GFFDense_256_U350_H600,
        epochs=10000,
        miniEpoch=200,
        lr=1e-4,
        train=train,
        verbose=verb,
    )

    return


# ##
# def run_mlp_training_nanofins(train=True, verb=False):
#     # run_training_neural_model(
#     #     model_caller=MLP_models.MLP_Nanofins_Dense32_U350_H600,
#     #     epochs=1,
#     #     miniEpoch=1,
#     #     lr=1e-4,
#     #     train=train,
#     #     verbose=verb,
#     # )
#     # run_training_neural_model(
#     #     mode8l_caller=MLP_models.MLP_Nanofins_Dense64_U350_H600,
#     #     epochs=1,
#     #     miniEpoch=1,
#     #     lr=1e-4,
#     #     train=train,
#     #     verbose=verb,
#     # )
#     # run_training_neural_model(
#     #     model_caller=MLP_models.MLP_Nanofins_Dense128_U350_H600,
#     #     epochs=1,
#     #     miniEpoch=1,
#     #     lr=1e-4,
#     #     train=train,
#     #     verbose=verb,
#     # )
#     run_training_neural_model(
#         model_caller=MLP_models.MLP_Nanofins_Dense256_U350_H600,
#         epochs=100000,
#         miniEpoch=1000,
#         lr=1e-4,
#         train=train,
#         verbose=verb,
#     )
#     # run_training_neural_model(
#     #     model_caller=MLP_models.MLP_Nanofins_Dense512_U350_H600,
#     #     epochs=1,
#     #     miniEpoch=1,
#     #     lr=1e-4,
#     #     train=train,
#     #     verbose=verb,
#     # )
#     # run_training_neural_model(
#     #     model_caller=MLP_models.MLP_Nanofins_Dense1024_U350_H600,
#     #     epochs=60000,
#     #     miniEpoch=1000,
#     #     lr=1e-4,
#     #     train=train,
#     #     verbose=verb,
#     # )

#     return


# def run_mlp_training_nanocylinders(train=True, verb=False):
#     run_training_neural_model(
#         model_caller=MLP_models.MLP_Nanocylinders_Dense256_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=MLP_models.MLP_Nanocylinders_Dense128_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=MLP_models.MLP_Nanocylinders_Dense64_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=MLP_models.MLP_Nanocylinders_Dense32_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         train=train,
#         verbose=verb,
#     )


# def run_ERBF_training_nanofins(train=True, verb=False):
#     # # Nanofins
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B128_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B256_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B512_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B1024_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B2048_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanofins_B4096_U350_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=10000,
#         train=train,
#         verbose=verb,
#     )

#     return


# def run_ERBF_training_nanocylinders(train=True, verb=False):
#     # # Nanocylinders
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanocylinders_B32_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanocylinders_B64_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanocylinders_B128_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanocylinders_B256_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )
#     run_training_neural_model(
#         model_caller=eRBF_models.ERBF_Nanocylinders_B512_U180_H600,
#         epochs=1,
#         miniEpoch=1,
#         batch_size=30000,
#         train=train,
#         verbose=verb,
#     )

#     return


if __name__ == "__main__":
    train_caller(train=True, verb=True)

    # run_mlp_training_nanofins(train=True, verb=True)
    # run_mlp_training_nanocylinders(train=True, verb=False)
    # run_ERBF_training_nanofins(train=True, verb=False)
    # run_ERBF_training_nanocylinders(train=True, verb=True)
