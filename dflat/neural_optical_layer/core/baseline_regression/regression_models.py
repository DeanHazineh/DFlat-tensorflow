import numpy as np
import time
import math
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from ..mlp_parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600


class multivariate_polynomial_regression_sklearn:
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
        complex_error = np.abs(
            pred_trans * np.exp(1j * pred_phase.numpy()) - true_trans * np.exp(1j * true_phase.numpy())
        )

        # get relative errors
        # rel_trans = trans_error / true_trans
        # rel_phase = phase_error / true_phase
        rel_complex = complex_error / np.abs(true_trans * np.exp(1j * true_phase.numpy()))

        # Estimate FLOPs
        est_FLOPs = self.profile_FLOPs()

        saveTo = self._modelSavePath + savestring
        data = {
            "trans_error": trans_error,
            # "rel_trans": rel_trans,
            "phase_error": phase_error,
            # "rel_phase": rel_phase,
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
        xtrain, xtest, ytrain, ytest = train_test_split(
            inputData, outputData, test_size=0.15, random_state=13, shuffle=True
        )

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


## Sub Models for comparison of MAE/FLOPs
class multipoly_nanocylinders_10(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_10")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_10/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(10)
        self.build_poly()


class multipoly_nanocylinders_12(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_12")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_12/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(12)
        self.build_poly()


class multipoly_nanocylinders_14(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_14")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_14/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(14)
        self.build_poly()


class multipoly_nanocylinders_16(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_16")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_16/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(16)
        self.build_poly()


class multipoly_nanocylinders_18(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_18")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_18/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(18)
        self.build_poly()


class multipoly_nanocylinders_20(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_20")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_20/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(20)
        self.build_poly()


class multipoly_nanocylinders_22(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_22")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_22/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(22)
        self.build_poly()


class multipoly_nanocylinders_24(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_24")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_24/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(24)
        self.build_poly()


class multipoly_nanocylinders_26(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_26")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_26/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(26)
        self.build_poly()


class multipoly_nanocylinders_28(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_28")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_28/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(28)
        self.build_poly()


class multipoly_nanocylinders_30(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_30")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_30/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(30)
        self.build_poly()


class multipoly_nanocylinders_32(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_32")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_32/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(32)
        self.build_poly()


class multipoly_nanocylinders_34(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanocylinders_U180_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanocylinders_U180_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanocylinders_34")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanocylinders_34/"
        )

        self._set_input_features(2)
        self._set_output_features(3)
        self._set_poly_degree(34)
        self.build_poly()


class multipoly_nanofins_6(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_6")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_6/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(6)
        self.build_poly()


class multipoly_nanofins_7(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_7")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_7/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(7)
        self.build_poly()


class multipoly_nanofins_8(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_8")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_8/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(8)
        self.build_poly()


class multipoly_nanofins_9(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_9")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_9/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(9)
        self.build_poly()


class multipoly_nanofins_10(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_10")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_10/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(10)
        self.build_poly()


class multipoly_nanofins_11(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_11")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_11/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(11)
        self.build_poly()


class multipoly_nanofins_12(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_12")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_12/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(12)
        self.build_poly()


class multipoly_nanofins_13(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_13")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_13/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(12)
        self.build_poly()


class multipoly_nanofins_14(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_14")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_14/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(14)
        self.build_poly()


class multipoly_nanofins_15(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_15")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_15/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(15)
        self.build_poly()


class multipoly_nanofins_16(
    multivariate_polynomial_regression_sklearn,
    MLP_Nanofins_U350_H600,
):
    def __init__(self, **kwargs):
        MLP_Nanofins_U350_H600.__init__(self)
        multivariate_polynomial_regression_sklearn.__init__(self)

        self.set_model_name("multipoly_nanofins_16")
        self.set_modelSavePath(
            "dflat/neural_optical_layer/core/baseline_regression/fitted_regression_models/multipoly_nanofins_16/"
        )

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(16)
        self.build_poly()
