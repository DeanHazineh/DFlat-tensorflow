import tensorflow as tf


def check_input_type(inputs, use_dtype):
    if not isinstance(inputs, list):
        inputs = list(inputs)

    modification_flag = False

    for idx, input_dat in enumerate(inputs):
        if not isinstance(input_dat, tf.Tensor):
            modification_flag = True
            try:
                inputs[idx] = tf.convert_to_tensor(input_dat, dtype=use_dtype)
            except ValueError:
                raise ValueError("Input data at index {} can't be converted to a tensorflow tensor".format(idx))
        elif input_dat.dtype != use_dtype:
            modification_flag = True
            inputs[idx] = tf.cast(input_dat, dtype=use_dtype)

    if len(inputs) > 1:
        return inputs, modification_flag
    else:
        return inputs[0], modification_flag


def check_broadband_wavelength_parameters(parameters):
    if not ("wavelength_set_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_set_m")

    return


def check_single_wavelength_parameters(parameters):
    if not ("wavelength_m" in parameters.keys()):
        raise KeyError("parameters must contain wavelength_m")

    if not (type(parameters["wavelength_m"]) == float):
        raise ValueError("Wavelength should be a single float")
    return
