import sys

sys.path.append(".")
from neural_optical_layer.core.baseline_regression.regression_models import *


def fit_regression(model_caller):
    model = model_caller()
    model.fit_polyCoeff()

    return


def run_regression_nanofins():
    fit_regression(model_caller=multipoly_nanofins_6)

    return


def run_regression_nanocylinders():
    fit_regression(model_caller=multipoly_nanocylinders_10)

    return


if __name__ == "__main__":
    run_regression_nanofins()
    run_regression_nanocylinders()
