import numpy as np
import tensorflow as tf


def convmat(A, P, Q):
    """
    This function computes a convolution matrix for a real space matrix `A` that
    represents either a relative permittivity or permeability distribution for a
    set of pixels, layers, and batch.
    Args:
        A: A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX,
        pixelsY, Nlayers, Nx, Ny)` specifying real space values on a Cartesian
        grid.

        P: A positive and odd `int` specifying the number of spatial harmonics
        along `T1`.

        Q: A positive and odd `int` specifying the number of spatial harmonics
        along `T2`.
    Returns:
        A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX,
        pixelsY, Nlayers, P * Q, P * Q)` representing a stack of convolution
        matrices based on `A`.
    """

    # Determine the shape of A.
    batchSize, pixelsX, pixelsY, Nlayers, Nx, Ny = A.shape

    # Compute indices of spatial harmonics.
    NH = P * Q  # total number of harmonics.
    p_max = np.floor(P / 2.0)
    q_max = np.floor(P / 2.0)

    # Indices along T1 and T2.
    p = np.linspace(-p_max, p_max, P)
    q = np.linspace(-q_max, q_max, Q)

    # Compute array indices of the center harmonic.
    p0 = int(np.floor(Nx / 2))
    q0 = int(np.floor(Ny / 2))

    # Fourier transform the real space distributions.
    A = tf.signal.fftshift(tf.signal.fft2d(A), axes=(4, 5)) / (Nx * Ny)

    # Build the matrix.
    firstCoeff = True
    for qrow in range(Q):
        for prow in range(P):
            for qcol in range(Q):
                for pcol in range(P):
                    pfft = int(p[prow] - p[pcol])
                    qfft = int(q[qrow] - q[qcol])

                    # Sequentially concatenate Fourier coefficients.
                    value = A[:, :, :, :, p0 + pfft, q0 + qfft]
                    value = value[:, :, :, :, tf.newaxis, tf.newaxis]
                    if firstCoeff:
                        firstCoeff = False
                        C = value
                    else:
                        C = tf.concat([C, value], axis=5)

    # Reshape the coefficients tensor into a stack of convolution matrices.
    convMatrixShape = (batchSize, pixelsX, pixelsY, Nlayers, P * Q, P * Q)
    matrixStack = tf.reshape(C, shape=convMatrixShape)

    return matrixStack


def redheffer_star_product(SA, SB):
    """
    This function computes the redheffer star product of two block matrices,
    which is the result of combining the S-parameter of two systems.
    Args:
        SA: A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of a system. `SA` needs to have the
        keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a `tf.Tensor`
        of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where NH is the
        total number of spatial harmonics.

        SB: A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of a second system. `SB` needs to have
        the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to a
        `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH)`, where
        NH is the total number of spatial harmonics.
    Returns:
        A `dict` of `tf.Tensor` values specifying the block matrix
        corresponding to the S-parameters of the combined system. `SA` needs
        to have the keys ('S11', 'S12', 'S21', 'S22'), where each key maps to
        a `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, 2*NH, 2*NH),
        where NH is the total number of spatial harmonics.
    """
    cdtype = SA["S11"].dtype

    # Define the identity matrix.
    batchSize, pixelsX, pixelsY, dim, _ = SA["S11"].shape
    I = tf.eye(num_rows=dim, dtype=cdtype)
    I = I[tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    I = tf.tile(I, multiples=(batchSize, pixelsX, pixelsY, 1, 1))

    # Calculate S11.
    S11 = tf.linalg.inv(I - tf.linalg.matmul(SB["S11"], SA["S22"]))
    S11 = tf.linalg.matmul(S11, SB["S11"])
    S11 = tf.linalg.matmul(SA["S12"], S11)
    S11 = SA["S11"] + tf.linalg.matmul(S11, SA["S21"])

    # Calculate S12.
    S12 = tf.linalg.inv(I - tf.linalg.matmul(SB["S11"], SA["S22"]))
    S12 = tf.linalg.matmul(S12, SB["S12"])
    S12 = tf.linalg.matmul(SA["S12"], S12)

    # Calculate S21.
    S21 = tf.linalg.inv(I - tf.linalg.matmul(SA["S22"], SB["S11"]))
    S21 = tf.linalg.matmul(S21, SA["S21"])
    S21 = tf.linalg.matmul(SB["S21"], S21)

    # Calculate S22.
    S22 = tf.linalg.inv(I - tf.linalg.matmul(SA["S22"], SB["S11"]))
    S22 = tf.linalg.matmul(S22, SA["S22"])
    S22 = tf.linalg.matmul(SB["S21"], S22)
    S22 = SB["S22"] + tf.linalg.matmul(S22, SB["S12"])

    # Store S parameters in an output dictionary.
    S = dict({})
    S["S11"] = S11
    S["S12"] = S12
    S["S21"] = S21
    S["S22"] = S22

    return S
