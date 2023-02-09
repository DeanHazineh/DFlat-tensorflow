import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def leakyrelu100(x):
    x_pos = (x + tf.abs(x)) / 2
    x_neg = (x - tf.abs(x)) / 2
    return x_pos + 0.01 * x_neg


def gaussian_activation(x, a):
    return tf.math.exp(-0.5 * x**2 / a**2)


class customLoss(tf.keras.losses.Loss):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.TF_ZERO = tf.constant(0.0, dtype=tf.float64)

    def call(self, y_true, y_pred):
        trans_true, phase_true = self.model.convert_output_complex(y_true)
        trans_model, phase_model = self.model.convert_output_complex(y_pred)

        field_true = tf.complex(trans_true, self.TF_ZERO) * tf.math.exp(tf.complex(self.TF_ZERO, phase_true))
        field_pred = tf.complex(trans_model, self.TF_ZERO) * tf.math.exp(tf.complex(self.TF_ZERO, phase_model))

        # MAE of Complex phasors
        return tf.math.reduce_mean(
            tf.math.sqrt(tf.math.square(tf.math.real(field_true - field_pred)) + tf.math.square(tf.math.imag(field_true - field_pred)))
        )


def get_flops_alternate(keras_sequential_model):
    concrete = tf.function(lambda inputs: keras_sequential_model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in keras_sequential_model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops
