import tensorflow as tf
from .arch_Core_class import Multivariate_Polynomial_Object
from .arch_Parent_class import MLP_Nanofins_U350_H600, MLP_Nanocylinders_U180_H600

multipoly_model_names = [
    # "multipoly_nanofins_6",
    # "multipoly_nanofins_12",
    # "multipoly_nanofins_18",
    # "multipoly_nanofins_24",
]

### nanofin multivariate polynomial
class multipoly_nanofins_6(Multivariate_Polynomial_Object, MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        # Super initialize the MLP_Nanofins Parent followed by multivariatePoly architecture
        MLP_Nanofins_U350_H600.__init__(self, dtype=dtype)
        Multivariate_Polynomial_Object.__init__(self)

        model_name = "multipoly_nanofins_6"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MultiPoly_models/" + model_name)
        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(6)
        self.build_poly()


class multipoly_nanofins_12(Multivariate_Polynomial_Object, MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        # Super initialize the MLP_Nanofins Parent followed by multivariatePoly architecture
        MLP_Nanofins_U350_H600.__init__(self, dtype=dtype)
        Multivariate_Polynomial_Object.__init__(self)

        model_name = "multipoly_nanofins_12"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MultiPoly_models/" + model_name)

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(12)
        self.build_poly()


class multipoly_nanofins_18(Multivariate_Polynomial_Object, MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        # Super initialize the MLP_Nanofins Parent followed by multivariatePoly architecture
        MLP_Nanofins_U350_H600.__init__(self, dtype=dtype)
        Multivariate_Polynomial_Object.__init__(self)

        model_name = "multipoly_nanofins_18"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MultiPoly_models/" + model_name)

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(18)
        self.build_poly()


class multipoly_nanofins_24(Multivariate_Polynomial_Object, MLP_Nanofins_U350_H600):
    def __init__(self, dtype=tf.float64):
        # Super initialize the MLP_Nanofins Parent followed by multivariatePoly architecture
        MLP_Nanofins_U350_H600.__init__(self, dtype=dtype)
        Multivariate_Polynomial_Object.__init__(self)

        model_name = "multipoly_nanofins_24"
        self.set_model_name(model_name)
        self.set_modelSavePath("trained_MultiPoly_models/" + model_name)

        self._set_input_features(3)
        self._set_output_features(6)
        self._set_poly_degree(24)
        self.build_poly()
