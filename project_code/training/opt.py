def split_3dmm_labels(labels):
    """
    split labels into different 3dmm params
    :param labels:
    :return:
    """
    # get different labels
    # Shape_Para: (199,)
    # Pose_Para: (7,)
    # Exp_Para: (29,)
    # Color_Para: (7,)
    # Illum_Para: (10,)
    # pt2d: (136, )
    # Tex_Para: (199,)

    shape_labels = labels[:, :199]
    pose_labels = labels[:, 199: 206]
    exp_labels = labels[:, 206: 235]
    color_labels = labels[:, 235: 242]
    illum_labels = labels[:, 242: 252]
    landmark_labels = labels[:, 252: 388]
    tex_labels = labels[:, 388:]

    return shape_labels, pose_labels, exp_labels, color_labels, illum_labels, landmark_labels, tex_labels


def compute_landmarks(pose, shape, output_size=224):
    # m: rotation matrix [batch_size x (4x2)]
    # shape: 3d vertices location [batch_size x (vertex_num x 3)]

    n_size = get_shape(pose)
    n_size = n_size[0]

    s = output_size

    # Tri, tri2vt
    kpts = load_3DMM_kpts()
    kpts_num = kpts.shape[0]

    indices = np.zeros([n_size, kpts_num, 2], np.int32)
    for i in range(n_size):
        indices[i, :, 0] = i
        indices[i, :, 1:2] = kpts

    indices = tf.constant(indices, tf.int32)

    kpts_const = tf.constant(kpts, tf.int32)

    vertex3d = tf.reshape(shape, shape=[n_size, -1, 3])  # batch_size x vertex_num x 3
    vertex3d = tf.gather_nd(vertex3d,
                            indices)  # Keypointd selection                                   # batch_size x kpts_num x 3
    vertex4d = tf.concat(axis=2, values=[vertex3d, tf.ones(get_shape(vertex3d)[0:2] + [1],
                                                           tf.float32)])  # batch_size x kpts_num x 4

    pose = tf.reshape(pose, shape=[n_size, 4, 2])
    vertex2d = tf.matmul(pose, vertex4d, True, True)  # batch_size x 2 x kpts_num
    vertex2d = tf.transpose(vertex2d, perm=[0, 2, 1])  # batch_size x kpts_num x 2

    [vertex2d_u, vertex2d_v] = tf.split(axis=2, num_or_size_splits=2, value=vertex2d)
    vertex2d_u = vertex2d_u - 1
    vertex2d_v = s - vertex2d_v

    return vertex2d_u, vertex2d_v