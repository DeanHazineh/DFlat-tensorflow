from dflat.unit_tests import *

testlist = [
    test_rcwa_layers,
    test_rcwa_layers_batched,
    test_neural_layer,
    test_response_to_param,
    test_fourier_layers_mono,
    test_fourier_layers_broadband,
]


def run_unit_tests():
    for fun in testlist:
        try:
            fun()
            print(fun.__name__, " TEST PASSED")
        except:
            print(fun.__name__, " TEST FAILED")

    return


if __name__ == "__main__":
    run_unit_tests()

