from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import math
from neural_optical_layer.core.mlp_Dense_models import MLP_Nanocylinders_U180_H600, MLP_Nanofins_U350_H600


def train_polyRep(modelClass, polyDegree, interactionOnly=False):
    # Load the input and output data, using same seed and split as the MLP
    model = modelClass
    inputData, outputData = model.returnLibraryAsTrainingData()
    xtrain, xtest, ytrain, ytest = train_test_split(
        inputData, outputData, test_size=0.15, random_state=13, shuffle=True
    )
    # The MLP had a validation set which pulled out 15% of training data
    # xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.15, random_state=13, shuffle=True)

    # Create Poly Object
    poly = PolynomialFeatures(polyDegree, interaction_only=interactionOnly)
    polyFeatures_train = poly.fit_transform(xtrain)
    polyFeatures_test = poly.fit_transform(xtest)

    holdPolyObjects = []
    for i in range(ytrain.shape[1]):
        polyModel = LinearRegression()
        polyModel.fit(polyFeatures_train, ytrain[:, i])
        holdPolyObjects.append(polyModel)

    # Evaluate on Test set
    poly_pred_test = eval_polyRep(holdPolyObjects, polyFeatures_test)
    pred_trans, pred_phase = model.convert_output_complex(poly_pred_test)
    true_trans, true_phase = model.convert_output_complex(ytest)
    pred_opt_resp = pred_trans * np.exp(1j * pred_phase.numpy())
    true_opt_resp = true_trans * np.exp(1j * true_phase.numpy())

    # Print Model Performance
    num_features = xtrain.shape[1]
    outSize = ytrain.shape[1]  # number of output variables of the model
    num_polyCoeffs_per_output = math.comb(num_features + polyDegree, polyDegree)
    MAE_test = np.mean(np.abs(pred_opt_resp - true_opt_resp))
    # FLOPs of matrix multiplication
    # (1 x num_polyCoeffs_per_output) _matmul_ (num_polyCoeffs_per_output x outsize)
    estFLOPs = 1 * outSize * (2 * num_polyCoeffs_per_output - 1)
    print("Degree: Params; Est FLOPs; MAE | ", polyDegree, num_polyCoeffs_per_output * outSize, estFLOPs, MAE_test)

    return holdPolyObjects, estFLOPs, MAE_test


def eval_polyRep(holdPolyObjects, polyFeatures):
    output = []
    for i in range(len(holdPolyObjects)):
        output.append(holdPolyObjects[i].predict(polyFeatures))

    return np.transpose(np.vstack(output), [1, 0])
