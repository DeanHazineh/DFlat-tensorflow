import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    print(
        "\n\nThis error most likely means that this notebook is not "
        "configured to use a GPU.  Change this in Notebook Settings via the "
        "command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n"
    )
    raise SystemError("GPU device not found")


def cpu():
    with tf.device("/cpu:0"):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device("/device:GPU:0"):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


def test_convolution_time():
    # We run each op once to warm up; see: https://stackoverflow.com/a/45067900
    cpu()
    gpu()

    # Run the op several times.
    print("Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images " "(batch x height x width x channel). Sum of ten runs.")
    print("CPU (s):")
    cpu_time = timeit.timeit("cpu()", number=10, setup="from __main__ import cpu")
    print(cpu_time)

    print("GPU (s):")
    gpu_time = timeit.timeit("gpu()", number=10, setup="from __main__ import gpu")
    print(gpu_time)
    print("GPU speedup over CPU: {}x".format(int(cpu_time / gpu_time)))

    return


if __name__ == "__main__":
    test_convolution_time()
