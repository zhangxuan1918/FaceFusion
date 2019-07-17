import tensorflow as tf
from tf_3dmm.mesh.transform import affine_transform

from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from tf_3dmm.tf_util import is_tf_expression


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
    n_size = labels.shape[0]
    shape_labels = labels[:, :199]
    pose_labels = labels[:, 199: 206]
    exp_labels = labels[:, 206: 235]
    color_labels = labels[:, 235: 242]
    illum_labels = labels[:, 242: 252]
    # reshape landmark
    landmark_labels = tf.reshape(labels[:, 252: 388], (-1, 2, 68))
    tex_labels = labels[:, 388:]

    return shape_labels, pose_labels, exp_labels, color_labels, illum_labels, landmark_labels, tex_labels


def _compute_landmarks_helper(shape_param, exp_param, pose_param, landmark_indices, bfm, input_image_size):
    """
    compute 2d landmarks from 3d landmarks

    1. get vertices from bfm models
    2. scale, rotate and translate the 3d vertices (weak prospective projection)
    3. flip y coordinates

    note, since the input frame size is 224 x 224, thus, the landmarks are computed based on this frame size

    :param vertices_3d: 3d vertices, [n_ver, 3]
    :param pose_param: pose parameters for scaling, rotation and translation, [1, 7]
    :param landmark_indices: indices for landmarks, [68, 1]
    :return: landmarks 2d, [68, 2]
    """

    vertices_3d = bfm.get_vertices(shape_param=shape_param, exp_param=exp_param)
    landmarks_3d = tf.gather_nd(vertices_3d, landmark_indices)
    landmarks_3d = affine_transform(
        vertices=landmarks_3d,
        scaling=pose_param[0, 6],
        angles_rad=pose_param[0, 0:3],
        t3d=pose_param[0, 3:6])

    landmarks_2d = tf.concat(
        [tf.reshape(landmarks_3d[:, 0], (-1, 1)), input_image_size - tf.reshape(landmarks_3d[:, 1], (-1, 1)) - 1], axis=1)
    return landmarks_2d


def compute_landmarks(poses_param, shapes_param, exps_param, bfm: TfMorphableModel, input_image_size=224):
    """
    compute landmarks using pose, shape and expression params

    :param poses_param: batch pose params. (batch_size, 199) = > (batch_size, 199, 1)
    :param shapes_param: batch shapes params. (batch_size, 7) = > (batch_size, 1, 7)
    :param exps_param: batch expression params. (batch_size, 29) = > (batch_size, 29, 1)
    :param bfm: 3dmm model
    :param input_image_size: the input size of face model, the pose params are computed with image of shape (input_image_size, input_image_size)
    :return: tensor: shape [batch_size, 2, 68]
    """

    assert is_tf_expression(poses_param)
    assert is_tf_expression(shapes_param)
    assert is_tf_expression(exps_param)

    # convert tensor to numpy array
    poses_param = tf.expand_dims(poses_param, axis=1)
    shapes_param = tf.expand_dims(shapes_param, axis=2)
    exps_param = tf.expand_dims(exps_param, axis=2)

    n_batch_size = poses_param.shape[0]

    tf.debugging.assert_shapes({
        poses_param: (n_batch_size, 199, 1),
        shapes_param: (n_batch_size, 1, 7),
        exps_param: (n_batch_size, 29, 1)
    })

    landmark_indices = tf.expand_dims(bfm.get_landmark_indices(), axis=1)
    n_landmarks = landmark_indices.shape()[0]
    landmarks = []
    for i in range(n_batch_size):
        pose = poses_param[i, :, :]
        shape = shapes_param[i, :, :]
        exp = exps_param[i, :, :]

        landmarks_2d = _compute_landmarks_helper(
            shape_param=shape,
            exp_param=exp,
            pose_param=pose,
            landmark_indices=landmark_indices,
            input_image_size=input_image_size,
            bfm=bfm
        )

        landmarks.append(tf.transpose(landmarks_2d))

    landmarks = tf.concat(landmarks, axis=0)

    tf.debugging.assert_shapes({landmarks: (n_batch_size, 2, n_landmarks)})

    return landmarks
