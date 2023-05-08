import numpy as np
import tensorflow as tf

from . import colburn_rcwa_utils as rcwa_utils
from . import colburn_tensor_utils as tensor_utils


def complex_pseudoinverse(tensor, rcond=1e-15):
    dtype = tensor.dtype
    if not dtype.is_complex:
        return tf.linalg.pinv(tensor)

    # Compute SVD
    s, u, v = tf.linalg.svd(tensor, full_matrices=False)

    # Compute the reciprocal of singular values
    cutoff = rcond * tf.reduce_max(s)
    s_inv = tf.where(s > cutoff, tf.math.reciprocal(s), 0.0)

    # Cast the reciprocal of singular values to complex
    s_inv_complex = tf.cast(s_inv, dtype)

    # Compute the pseudoinverse
    pseudo_inv_s = tf.linalg.diag(s_inv_complex)
    pseudo_inv = tf.matmul(v, tf.matmul(pseudo_inv_s, u, adjoint_b=True))

    return pseudo_inv


def batch_regularized_inverse(matrix, alpha=0.1):
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("The last two dimensions of the input tensor must be square.")

    matrix_shape = matrix.shape[:-2]
    identity = tf.eye(matrix.shape[-1], dtype=matrix.dtype)
    identity = tf.broadcast_to(identity, matrix_shape + matrix.shape[-2:])
    regularized_matrix = matrix + alpha * identity
    return complex_pseudoinverse(regularized_matrix)


def simulate(ER_t, UR_t, params):
    """
    Calculates the transmission/reflection coefficients for a unit cell with a
    given permittivity/permeability distribution and the batch of input conditions
    (e.g., wavelengths, wavevectors, polarizations) for a fixed real space grid
    and number of Fourier harmonics.

    Args:
        ER_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `tf.complex64` specifying the relative permittivity distribution
        of the unit cell.

        UR_t: A `tf.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `tf.complex64` specifying the relative permeability distribution
        of the unit cell.

        params: A `dict` containing simulation and optimization settings.
    Returns:
        outputs: A `dict` containing the keys {'rx', 'ry', 'rz', 'R', 'ref',
        'tx', 'ty', 'tz', 'T', 'TRN'} corresponding to the computed reflection/tranmission
        coefficients and powers. tx has shape [lambda, pixelsX, pixelsY, PQ, (?)]
    """

    # Extract commonly used parameters from the `params` dictionary.
    batchSize = params["batchSize"]
    pixelsX = params["pixelsX"]
    pixelsY = params["pixelsY"]
    Nlay = params["Nlay"]
    PQ = params["PQ"]
    dtype = params["dtype"]
    cdtype = params["cdtype"]

    ### Step 3: Build convolution matrices for the permittivity and permeability ###
    ERC = rcwa_utils.convmat(ER_t, PQ[0], PQ[1])
    URC = rcwa_utils.convmat(UR_t, PQ[0], PQ[1])

    ### Step 4: Wave vector expansion ###
    I = np.eye(np.prod(PQ), dtype=complex)
    I = tf.convert_to_tensor(I, dtype=cdtype)
    I = I[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    I = tf.tile(I, multiples=(batchSize, pixelsX, pixelsY, Nlay, 1, 1))

    Z = np.zeros((np.prod(PQ), np.prod(PQ)), dtype=complex)
    Z = tf.convert_to_tensor(Z, dtype=cdtype)
    Z = Z[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, :]
    Z = tf.tile(Z, multiples=(batchSize, pixelsX, pixelsY, Nlay, 1, 1))

    n1 = tf.math.sqrt(params["er1"])
    n2 = tf.math.sqrt(params["er2"])

    k0 = tf.cast(2 * np.pi / params["lam0"], dtype=cdtype)
    # kinc_x0 = tf.cast(n1 * tf.sin(params["theta"]) * tf.cos(params["phi"]), dtype=cdtype)
    # kinc_y0 = tf.cast(n1 * tf.sin(params["theta"]) * tf.sin(params["phi"]), dtype=cdtype)
    # kinc_z0 = tf.cast(n1 * tf.cos(params["theta"]), dtype=cdtype)
    kinc_x0 = n1 * tf.cast(tf.sin(params["theta"]) * tf.cos(params["phi"]), dtype=cdtype)
    kinc_y0 = n1 * tf.cast(tf.sin(params["theta"]) * tf.sin(params["phi"]), dtype=cdtype)
    kinc_z0 = n1 * tf.cast(tf.cos(params["theta"]), dtype=cdtype)
    kinc_z0 = kinc_z0[:, :, :, 0, :, :]

    # Unit vectors
    T1 = np.transpose([2 * np.pi / params["Lx"], 0])
    T2 = np.transpose([0, 2 * np.pi / params["Ly"]])
    p_max = np.floor(PQ[0] / 2.0)
    q_max = np.floor(PQ[1] / 2.0)
    p = tf.constant(np.linspace(-p_max, p_max, PQ[0]), dtype=cdtype)  # indices along T1
    p = p[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    p = tf.tile(p, multiples=(1, pixelsX, pixelsY, Nlay, 1, 1))
    q = tf.constant(np.linspace(-q_max, q_max, PQ[1]), dtype=cdtype)  # indices along T2
    q = q[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
    q = tf.tile(q, multiples=(1, pixelsX, pixelsY, Nlay, 1, 1))

    # Build Kx and Ky matrices
    kx_zeros = tf.zeros(PQ[1], dtype=cdtype)
    kx_zeros = kx_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :]
    ky_zeros = tf.zeros(PQ[0], dtype=cdtype)
    ky_zeros = ky_zeros[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    kx = kinc_x0 - 2 * np.pi * p / (k0 * params["Lx"]) - kx_zeros
    ky = kinc_y0 - 2 * np.pi * q / (k0 * params["Ly"]) - ky_zeros

    kx_T = tf.transpose(kx, perm=[0, 1, 2, 3, 5, 4])
    KX = tf.reshape(kx_T, shape=(batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
    KX = tf.linalg.diag(KX)

    ky_T = tf.transpose(ky, perm=[0, 1, 2, 3, 5, 4])
    KY = tf.reshape(ky_T, shape=(batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ)))
    KY = tf.linalg.diag(KY)

    KZref = tf.linalg.matmul(tf.math.conj(params["ur1"] * I), tf.math.conj(params["er1"] * I))
    KZref = KZref - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZref = tf.math.sqrt(KZref)
    KZref = -tf.math.conj(KZref)

    KZtrn = tf.linalg.matmul(tf.math.conj(params["ur2"] * I), tf.math.conj(params["er2"] * I))
    KZtrn = KZtrn - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZtrn = tf.math.sqrt(KZtrn)
    KZtrn = tf.math.conj(KZtrn)

    ### Step 5: Free Space ###
    KZ = I - tf.linalg.matmul(KX, KX) - tf.linalg.matmul(KY, KY)
    KZ = tf.math.sqrt(KZ)
    KZ = tf.math.conj(KZ)

    Q_free_00 = tf.linalg.matmul(KX, KY)
    Q_free_01 = I - tf.linalg.matmul(KX, KX)
    Q_free_10 = tf.linalg.matmul(KY, KY) - I
    Q_free_11 = -tf.linalg.matmul(KY, KX)
    Q_free_row0 = tf.concat([Q_free_00, Q_free_01], axis=5)
    Q_free_row1 = tf.concat([Q_free_10, Q_free_11], axis=5)
    Q_free = tf.concat([Q_free_row0, Q_free_row1], axis=4)

    W0_row0 = tf.concat([I, Z], axis=5)
    W0_row1 = tf.concat([Z, I], axis=5)
    W0 = tf.concat([W0_row0, W0_row1], axis=4)

    LAM_free_row0 = tf.concat([1j * KZ, Z], axis=5)
    LAM_free_row1 = tf.concat([Z, 1j * KZ], axis=5)
    LAM_free = tf.concat([LAM_free_row0, LAM_free_row1], axis=4)

    V0 = tf.linalg.matmul(Q_free, tf.linalg.inv(LAM_free))

    ### Step 6: Initialize Global Scattering Matrix ###
    SG = dict({})
    SG_S11 = tf.zeros(shape=(2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=cdtype)
    SG["S11"] = tensor_utils.expand_and_tile_tf(SG_S11, batchSize, pixelsX, pixelsY)

    SG_S12 = tf.eye(num_rows=2 * np.prod(PQ), dtype=cdtype)
    SG["S12"] = tensor_utils.expand_and_tile_tf(SG_S12, batchSize, pixelsX, pixelsY)

    SG_S21 = tf.eye(num_rows=2 * np.prod(PQ), dtype=cdtype)
    SG["S21"] = tensor_utils.expand_and_tile_tf(SG_S21, batchSize, pixelsX, pixelsY)

    SG_S22 = tf.zeros(shape=(2 * np.prod(PQ), 2 * np.prod(PQ)), dtype=cdtype)
    SG["S22"] = tensor_utils.expand_and_tile_tf(SG_S22, batchSize, pixelsX, pixelsY)

    ### Step 7: Calculate eigenmodes ###

    # Build the eigenvalue problem.
    P_00 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
    # P_00 = tf.linalg.matmul(KX, batch_regularized_inverse(ERC))
    P_00 = tf.linalg.matmul(P_00, KY)

    P_01 = tf.linalg.matmul(KX, tf.linalg.inv(ERC))
    # P_01 = tf.linalg.matmul(KX, batch_regularized_inverse(ERC))
    P_01 = tf.linalg.matmul(P_01, KX)
    P_01 = URC - P_01

    P_10 = tf.linalg.matmul(KY, tf.linalg.inv(ERC))
    # P_10 = tf.linalg.matmul(KY, batch_regularized_inverse(ERC))
    P_10 = tf.linalg.matmul(P_10, KY) - URC

    P_11 = tf.linalg.matmul(-KY, tf.linalg.inv(ERC))
    # P_11 = tf.linalg.matmul(-KY, batch_regularized_inverse(ERC))
    P_11 = tf.linalg.matmul(P_11, KX)

    P_row0 = tf.concat([P_00, P_01], axis=5)
    P_row1 = tf.concat([P_10, P_11], axis=5)
    P = tf.concat([P_row0, P_row1], axis=4)

    Q_00 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
    Q_00 = tf.linalg.matmul(Q_00, KY)

    Q_01 = tf.linalg.matmul(KX, tf.linalg.inv(URC))
    Q_01 = tf.linalg.matmul(Q_01, KX)
    Q_01 = ERC - Q_01

    Q_10 = tf.linalg.matmul(KY, tf.linalg.inv(URC))
    Q_10 = tf.linalg.matmul(Q_10, KY) - ERC

    Q_11 = tf.linalg.matmul(-KY, tf.linalg.inv(URC))
    Q_11 = tf.linalg.matmul(Q_11, KX)

    Q_row0 = tf.concat([Q_00, Q_01], axis=5)
    Q_row1 = tf.concat([Q_10, Q_11], axis=5)
    Q = tf.concat([Q_row0, Q_row1], axis=4)

    # Compute eignmodes for the layers in each pixel for the whole batch.
    OMEGA_SQ = tf.linalg.matmul(P, Q)
    LAM, W = tensor_utils.eig_general(OMEGA_SQ)
    LAM = tf.sqrt(LAM)
    LAM = tf.linalg.diag(LAM)

    V = tf.linalg.matmul(Q, W)
    V = tf.linalg.matmul(V, tf.linalg.inv(LAM))

    # Scattering matrices for the layers in each pixel for the whole batch.
    W_inv = tf.linalg.inv(W)
    V_inv = tf.linalg.inv(V)
    A = tf.linalg.matmul(W_inv, W0) + tf.linalg.matmul(V_inv, V0)
    B = tf.linalg.matmul(W_inv, W0) - tf.linalg.matmul(V_inv, V0)

    X = tf.linalg.expm(-LAM * k0 * params["L"])

    S = dict({})
    A_inv = tf.linalg.inv(A)
    S11_left = tf.linalg.matmul(X, B)
    S11_left = tf.linalg.matmul(S11_left, A_inv)
    S11_left = tf.linalg.matmul(S11_left, X)
    S11_left = tf.linalg.matmul(S11_left, B)
    S11_left = A - S11_left
    S11_left = tf.linalg.inv(S11_left)

    S11_right = tf.linalg.matmul(X, B)
    S11_right = tf.linalg.matmul(S11_right, A_inv)
    S11_right = tf.linalg.matmul(S11_right, X)
    S11_right = tf.linalg.matmul(S11_right, A)
    S11_right = S11_right - B
    S["S11"] = tf.linalg.matmul(S11_left, S11_right)

    S12_right = tf.linalg.matmul(B, A_inv)
    S12_right = tf.linalg.matmul(S12_right, B)
    S12_right = A - S12_right
    S12_left = tf.linalg.matmul(S11_left, X)
    S["S12"] = tf.linalg.matmul(S12_left, S12_right)

    S["S21"] = S["S12"]
    S["S22"] = S["S11"]

    # Update the global scattering matrices.
    for l in range(Nlay):
        S_layer = dict({})
        S_layer["S11"] = S["S11"][:, :, :, l, :, :]
        S_layer["S12"] = S["S12"][:, :, :, l, :, :]
        S_layer["S21"] = S["S21"][:, :, :, l, :, :]
        S_layer["S22"] = S["S22"][:, :, :, l, :, :]
        SG = rcwa_utils.redheffer_star_product(SG, S_layer)

    ### Step 8: Reflection side ###
    # Eliminate layer dimension for tensors as they are unchanging on this dimension.
    KX = KX[:, :, :, 0, :, :]
    KY = KY[:, :, :, 0, :, :]
    KZref = KZref[:, :, :, 0, :, :]
    KZtrn = KZtrn[:, :, :, 0, :, :]
    Z = Z[:, :, :, 0, :, :]
    I = I[:, :, :, 0, :, :]
    W0 = W0[:, :, :, 0, :, :]
    V0 = V0[:, :, :, 0, :, :]
    ur1_red = params["ur1"][:, :, :, 0, :, :]
    ur2_red = params["ur2"][:, :, :, 0, :, :]
    er1_red = params["er1"][:, :, :, 0, :, :]
    er2_red = params["er2"][:, :, :, 0, :, :]

    Q_ref_00 = tf.linalg.matmul(KX, KY)
    # Q_ref_01 = params["ur1"] * params["er1"] * I - tf.linalg.matmul(KX, KX)
    Q_ref_01 = ur1_red * er1_red * I - tf.linalg.matmul(KX, KX)
    Q_ref_10 = tf.linalg.matmul(KY, KY) - ur1_red * er1_red * I
    Q_ref_11 = -tf.linalg.matmul(KY, KX)

    Q_ref_row0 = tf.concat([Q_ref_00, Q_ref_01], axis=4)
    Q_ref_row1 = tf.concat([Q_ref_10, Q_ref_11], axis=4)
    Q_ref = tf.concat([Q_ref_row0, Q_ref_row1], axis=3)

    W_ref_row0 = tf.concat([I, Z], axis=4)
    W_ref_row1 = tf.concat([Z, I], axis=4)
    W_ref = tf.concat([W_ref_row0, W_ref_row1], axis=3)

    LAM_ref_row0 = tf.concat([-1j * KZref, Z], axis=4)
    LAM_ref_row1 = tf.concat([Z, -1j * KZref], axis=4)
    LAM_ref = tf.concat([LAM_ref_row0, LAM_ref_row1], axis=3)

    V_ref = tf.linalg.matmul(Q_ref, tf.linalg.inv(LAM_ref))

    W0_inv = tf.linalg.inv(W0)
    V0_inv = tf.linalg.inv(V0)
    A_ref = tf.linalg.matmul(W0_inv, W_ref) + tf.linalg.matmul(V0_inv, V_ref)
    A_ref_inv = tf.linalg.inv(A_ref)
    B_ref = tf.linalg.matmul(W0_inv, W_ref) - tf.linalg.matmul(V0_inv, V_ref)

    SR = dict({})
    SR["S11"] = tf.linalg.matmul(-A_ref_inv, B_ref)
    SR["S12"] = 2 * A_ref_inv
    SR_S21 = tf.linalg.matmul(B_ref, A_ref_inv)
    SR_S21 = tf.linalg.matmul(SR_S21, B_ref)
    SR["S21"] = 0.5 * (A_ref - SR_S21)
    SR["S22"] = tf.linalg.matmul(B_ref, A_ref_inv)

    ### Step 9: Transmission side ###
    Q_trn_00 = tf.linalg.matmul(KX, KY)
    Q_trn_01 = ur2_red * er2_red * I - tf.linalg.matmul(KX, KX)
    Q_trn_10 = tf.linalg.matmul(KY, KY) - ur2_red * er2_red * I
    Q_trn_11 = -tf.linalg.matmul(KY, KX)
    Q_trn_row0 = tf.concat([Q_trn_00, Q_trn_01], axis=4)
    Q_trn_row1 = tf.concat([Q_trn_10, Q_trn_11], axis=4)
    Q_trn = tf.concat([Q_trn_row0, Q_trn_row1], axis=3)

    W_trn_row0 = tf.concat([I, Z], axis=4)
    W_trn_row1 = tf.concat([Z, I], axis=4)
    W_trn = tf.concat([W_trn_row0, W_trn_row1], axis=3)

    LAM_trn_row0 = tf.concat([1j * KZtrn, Z], axis=4)
    LAM_trn_row1 = tf.concat([Z, 1j * KZtrn], axis=4)
    LAM_trn = tf.concat([LAM_trn_row0, LAM_trn_row1], axis=3)

    V_trn = tf.linalg.matmul(Q_trn, tf.linalg.inv(LAM_trn))

    W0_inv = tf.linalg.inv(W0)
    V0_inv = tf.linalg.inv(V0)
    A_trn = tf.linalg.matmul(W0_inv, W_trn) + tf.linalg.matmul(V0_inv, V_trn)
    A_trn_inv = tf.linalg.inv(A_trn)
    B_trn = tf.linalg.matmul(W0_inv, W_trn) - tf.linalg.matmul(V0_inv, V_trn)

    ST = dict({})
    ST["S11"] = tf.linalg.matmul(B_trn, A_trn_inv)
    ST_S12 = tf.linalg.matmul(B_trn, A_trn_inv)
    ST_S12 = tf.linalg.matmul(ST_S12, B_trn)
    ST["S12"] = 0.5 * (A_trn - ST_S12)
    ST["S21"] = 2 * A_trn_inv
    ST["S22"] = tf.linalg.matmul(-A_trn_inv, B_trn)

    ### Step 10: Compute global scattering matrix ###
    SG = rcwa_utils.redheffer_star_product(SR, SG)
    SG = rcwa_utils.redheffer_star_product(SG, ST)

    ### Step 11: Compute source parameters ###

    # Compute mode coefficients of the source.
    delta = np.zeros((batchSize, pixelsX, pixelsY, np.prod(PQ)))
    delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1

    # Incident wavevector.
    kinc_x0_pol = tf.math.real(kinc_x0[:, :, :, 0, 0])
    kinc_y0_pol = tf.math.real(kinc_y0[:, :, :, 0, 0])
    kinc_z0_pol = tf.math.real(kinc_z0[:, :, :, 0])
    kinc_pol = tf.concat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], axis=3)

    # Calculate TE and TM polarization unit vectors.
    firstPol = True
    for pol in range(batchSize):
        if kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0:
            ate_pol = np.zeros((1, pixelsX, pixelsY, 3))
            ate_pol[:, :, :, 1] = 1
            ate_pol = tf.convert_to_tensor(ate_pol, dtype=dtype)
        else:
            # Calculation of `ate` for oblique incidence.
            n_hat = np.zeros((1, pixelsX, pixelsY, 3))
            n_hat[:, :, :, 0] = 1
            n_hat = tf.convert_to_tensor(n_hat, dtype=dtype)
            kinc_pol_iter = kinc_pol[pol, :, :, :]
            kinc_pol_iter = kinc_pol_iter[tf.newaxis, :, :, :]
            ate_cross = tf.linalg.cross(n_hat, kinc_pol_iter)
            ate_pol = ate_cross / tf.norm(ate_cross, axis=3, keepdims=True)

        if firstPol:
            ate = ate_pol
            firstPol = False
        else:
            ate = tf.concat([ate, ate_pol], axis=0)

    atm_cross = tf.linalg.cross(kinc_pol, ate)
    atm = atm_cross / tf.norm(atm_cross, axis=3, keepdims=True)
    ate = tf.cast(ate, dtype=cdtype)
    atm = tf.cast(atm, dtype=cdtype)

    # Decompose the TE and TM polarization into x and y components.
    EP = params["pte"] * ate + params["ptm"] * atm
    EP_x = EP[:, :, :, 0]
    EP_x = EP_x[:, :, :, tf.newaxis]
    EP_y = EP[:, :, :, 1]
    EP_y = EP_y[:, :, :, tf.newaxis]

    esrc_x = EP_x * delta
    esrc_y = EP_y * delta
    esrc = tf.concat([esrc_x, esrc_y], axis=3)
    esrc = esrc[:, :, :, :, tf.newaxis]

    W_ref_inv = tf.linalg.inv(W_ref)

    ### Step 12: Compute reflected and transmitted fields ###
    csrc = tf.linalg.matmul(W_ref_inv, esrc)

    # Compute tranmission and reflection mode coefficients.
    cref = tf.linalg.matmul(SG["S11"], csrc)
    ctrn = tf.linalg.matmul(SG["S21"], csrc)
    eref = tf.linalg.matmul(W_ref, cref)
    etrn = tf.linalg.matmul(W_trn, ctrn)

    rx = eref[:, :, :, 0 : np.prod(PQ), :]
    ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
    tx = etrn[:, :, :, 0 : np.prod(PQ), :]
    ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]

    # Compute longitudinal components.
    KZref_inv = tf.linalg.inv(KZref)
    KZtrn_inv = tf.linalg.inv(KZtrn)
    rz = tf.linalg.matmul(KX, rx) + tf.linalg.matmul(KY, ry)
    rz = tf.linalg.matmul(-KZref_inv, rz)
    tz = tf.linalg.matmul(KX, tx) + tf.linalg.matmul(KY, ty)
    tz = tf.linalg.matmul(-KZtrn_inv, tz)

    ### Step 13: Compute diffraction efficiences ###
    rx2 = tf.math.real(rx) ** 2 + tf.math.imag(rx) ** 2
    ry2 = tf.math.real(ry) ** 2 + tf.math.imag(ry) ** 2
    rz2 = tf.math.real(rz) ** 2 + tf.math.imag(rz) ** 2
    R2 = rx2 + ry2 + rz2

    R = tf.math.real(-KZref / ur1_red) / tf.math.real(kinc_z0 / ur1_red)
    R = tf.linalg.matmul(R, R2)
    R = tf.reshape(R, shape=(batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
    REF = tf.math.reduce_sum(R, axis=[3, 4])

    tx2 = tf.math.real(tx) ** 2 + tf.math.imag(tx) ** 2
    ty2 = tf.math.real(ty) ** 2 + tf.math.imag(ty) ** 2
    tz2 = tf.math.real(tz) ** 2 + tf.math.imag(tz) ** 2
    T2 = tx2 + ty2 + tz2
    T = tf.math.real(KZtrn / ur2_red) / tf.math.real(kinc_z0 / ur2_red)
    T = tf.linalg.matmul(T, T2)
    T = tf.reshape(T, shape=(batchSize, pixelsX, pixelsY, PQ[0], PQ[1]))
    TRN = tf.math.reduce_sum(T, axis=[3, 4])

    # Store the transmission/reflection coefficients and powers in a dictionary.
    outputs = dict({})
    outputs["rx"] = rx
    outputs["ry"] = ry
    outputs["rz"] = rz
    outputs["R"] = R
    outputs["REF"] = REF
    outputs["tx"] = tx
    outputs["ty"] = ty
    outputs["tz"] = tz
    outputs["T"] = T
    outputs["TRN"] = TRN

    return outputs
