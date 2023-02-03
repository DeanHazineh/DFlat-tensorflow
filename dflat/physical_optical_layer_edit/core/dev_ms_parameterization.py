
# def build_nine_rectangle_pattern(norm_param, span_limits, Lx, Ly, x_mesh, y_mesh, sigmoid_coeff, Nlay):
#     TF_ZERO = tf.constant(0.0, dtype=tf.float32)

#     dLx = Lx / 4
#     dLy = Ly / 4
#     cidx_x = [-dLx, 0, dLx, -dLx, 0, dLx, -dLx, 0, dLx]
#     cidx_y = [-dLy, -dLy, -dLy, 0, 0, 0, dLy, dLy, dLy]

#     struct_meta = []
#     for shape_idx in range(norm_param.shape[3]):
#         unit_shape = norm_param[:, :, :, shape_idx : shape_idx + 1]
#         unit_binary = build_rectangle_resonator(
#             unit_shape,
#             span_limits,
#             Lx,
#             Ly,
#             x_mesh + cidx_x[shape_idx],
#             y_mesh + cidx_y[shape_idx],
#             sigmoid_coeff,
#             Nlay,
#         )
#         # struct_meta.append(tf.math.abs(unit_binary[1]))
#         struct_meta.append(unit_binary[1])

#     struct_meta = tf.math.reduce_sum(tf.stack(struct_meta), 0)
#     # additional care needs to be taken for overlapping squares
#     # struct_meta = struct_meta - tf.constant(0.25, dtype=tf.float32)
#     # struct_meta = tf.complex(tf.math.sigmoid(sigmoid_coeff * struct_meta), TF_ZERO)
#     # struct_meta = tf.complex(struct_meta, TF_ZERO)

#     struct_binaries = []
#     for i in range(Nlay):
#         struct_binaries.append(None)
#     struct_binaries[1] = struct_meta

#     return struct_binaries
