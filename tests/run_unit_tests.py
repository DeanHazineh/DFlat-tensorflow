from dflat.tests_unit import *

testlist = [
    test_rcwa_layers,
    test_rcwa_layers_batched,
    test_neural_layer,
    test_response_to_param,
    test_fourier_layers_mono,
    test_fourier_layers_broadband,
    test_open_meta_libraries,
]


def run_unit_tests():
    for fun in testlist:
        try:
            fun()
            print("\033[91m" + fun.__name__, " TEST PASSED" + "\033[0m")

        except:
            print("\033[31m" + fun.__name__ + " TEST FAILED" + "\033[0m")

    return


if __name__ == "__main__":
    run_unit_tests()
