from dflat.neural_optical_layer.core.models_multivariate_polynomial import *


def fit_regression(model_caller):
    model = model_caller()
    model.fit_polyCoeff()

    return


def run_regression():
    # fit_regression(model_caller=multipoly_nanofins_6)
    # fit_regression(model_caller=multipoly_nanofins_12)
    # fit_regression(model_caller=multipoly_nanofins_18)
    fit_regression(model_caller=multipoly_nanofins_24)

    return


if __name__ == "__main__":
    run_regression()
