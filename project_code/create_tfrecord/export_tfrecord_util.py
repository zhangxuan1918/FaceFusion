import sys

import scipy.io as sio
import numpy as np
import tensorflow as tf
import PIL


def fn_extract_300W_LP_labels(param_mean_std_path, image_size, is_aflw_2000=False):
    # Load label mean and std
    param_mean_std = np.load(param_mean_std_path)

    shape_avg = param_mean_std['Shape_Para_mean']
    shape_std = param_mean_std['Shape_Para_std'] + 0.000000000000000001
    pose_avg = param_mean_std['Pose_Para_mean']
    pose_std = param_mean_std['Pose_Para_std'] + 0.000000000000000001
    exp_avg = param_mean_std['Exp_Para_mean']
    exp_std = param_mean_std['Exp_Para_std'] + 0.000000000000000001
    color_avg = param_mean_std['Color_Para_mean']
    color_std = param_mean_std['Color_Para_std'] + 0.000000000000000001
    illum_avg = param_mean_std['Illum_Para_mean']
    illum_std = param_mean_std['Illum_Para_std'] + 0.000000000000000001
    tex_avg = param_mean_std['Tex_Para_mean']
    tex_std = param_mean_std['Tex_Para_std'] + 0.000000000000000001

    image_size = 1.0 * image_size

    def get_labels(img_filename):
        # label file has the same name as image
        # mat format
        # roi: shape=(1, 4)
        # Shape_Para: shape=(199, 1)
        # Pose_Para: shape=(1, 7)
        # Exp_Para: shape=(29, 1)
        # Color_Para: shape=(1, 6): remove last value as it's always 1
        # Illum_Para: shape=(1, 9): remove last value as it's always 20
        # pt2d: shape=(2, 68)
        # Tex_Para: shape=(40, 1)
        # Total: 430 params

        mat_filename = '.'.join(img_filename.split('.')[:-1]) + '.mat'
        mat = sio.loadmat(mat_filename)

        # normalize pose param
        pp = mat['Pose_Para']
        pp[0, 3:] /= image_size
        pp = (pp - pose_avg) / pose_std

        # normalize landmarks
        if is_aflw_2000:
            lm = mat['pt3d_68'][:2, :] / image_size
        else:
            lm = mat['pt2d'] / image_size

        lm = np.reshape(lm, (-1,))  # to recover lm, np.reshape(lm, (2, -1))

        # update roi
        roi = mat['roi'] / image_size

        # rescale shape, exp and tex by their eigenvalue
        shape_para = (mat['Shape_Para'] - shape_avg) / shape_std
        exp_para = (mat['Exp_Para'] - exp_avg) / exp_std
        tex_para = (mat['Tex_Para'][:40, :] - tex_avg) / tex_std

        color_para = (mat['Color_Para'][:, :6] - color_avg) / color_std
        illum_para = (mat['Illum_Para'][:, :9] - illum_avg) / illum_std
        return np.concatenate((roi, lm, pp, shape_para, exp_para, color_para, illum_para, tex_para), axis=None)

    return get_labels


def split_300W_LP_labels(labels):
    # roi: shape=(1, 4)
    # Shape_Para: shape=(199, 1)
    # Pose_Para: shape=(1, 7)
    # Exp_Para: shape=(29, 1)
    # Color_Para: shape=(1, 6): remove last value as it's always 1
    # Illum_Para: shape=(1, 9): remove last value as it's always 2
    # pt2d: shape=(2, 68)
    # Tex_Para: shape=(40, 1)
    # Total: 430 params
    """

    :param labels:
    :return:

    roi: shape=(None, 4)
    Shape_Para: shape=(None, 199)
    Pose_Para: shape=(None, 7)
    Exp_Para: shape=(None, 29)
    Color_Para: shape=(None, 6): remove last value as it's always 1
    Illum_Para: shape=(None, 9): remove last value as it's always 2
    pt2d: shape=(None, 136)
    Tex_Para: shape=(None, 40)
    """
    assert isinstance(labels, tf.Tensor)
    if labels.shape[1] == 426:
        # without roi
        # with landmarks
        lm, pp, shape_para, exp_para, color_para, illum_para, tex_para = \
            tf.split(labels, num_or_size_splits=[136, 7, 199, 29, 6, 9, 40], axis=1)
        return None, lm, pp, shape_para, exp_para, color_para, illum_para, tex_para
    elif labels.shape[1] == 430:
        # with roi
        # with landmarks
        roi, lm, pp, shape_para, exp_para, color_para, illum_para, tex_para = \
            tf.split(labels, num_or_size_splits=[4, 136, 7, 199, 29, 6, 9, 40], axis=1)
        return roi, lm, pp, shape_para, exp_para, color_para, illum_para, tex_para
    elif labels.shape[1] == 290:
        # without roi
        # without landmarks
        pp, shape_para, exp_para, color_para, illum_para, tex_para = \
            tf.split(labels, num_or_size_splits=[7, 199, 29, 6, 9, 40], axis=1)
        return None, None, pp, shape_para, exp_para, color_para, illum_para, tex_para
    else:
        raise Exception('`labels` shape[1] is wrong')


def fn_unnormalize_300W_LP_labels(param_mean_std_path, image_size):
    # Load label mean and std
    param_mean_std = np.load(param_mean_std_path)

    shape_avg = tf.constant(param_mean_std['Shape_Para_mean'], dtype=tf.float32)
    shape_avg = tf.transpose(shape_avg, [1, 0])
    shape_std = tf.constant(param_mean_std['Shape_Para_std'] + 0.00000000000000000, dtype=tf.float32)
    shape_std = tf.transpose(shape_std, [1, 0])

    pose_avg = tf.constant(param_mean_std['Pose_Para_mean'], dtype=tf.float32)
    pose_std = tf.constant(param_mean_std['Pose_Para_std'] + 0.000000000000000001, dtype=tf.float32)

    exp_avg = tf.constant(param_mean_std['Exp_Para_mean'], dtype=tf.float32)
    exp_avg = tf.transpose(exp_avg, [1, 0])
    exp_std = tf.constant(param_mean_std['Exp_Para_std'] + 0.000000000000000001, dtype=tf.float32)
    exp_std = tf.transpose(exp_std, [1, 0])

    color_avg = tf.constant(param_mean_std['Color_Para_mean'], dtype=tf.float32)
    color_std = tf.constant(param_mean_std['Color_Para_std'] + 0.000000000000000001, dtype=tf.float32)
    illum_avg = tf.constant(param_mean_std['Illum_Para_mean'], dtype=tf.float32)
    illum_std = tf.constant(param_mean_std['Illum_Para_std'] + 0.000000000000000001, dtype=tf.float32)

    tex_avg = tf.constant(param_mean_std['Tex_Para_mean'], dtype=tf.float32)
    tex_avg = tf.transpose(tex_avg, [1, 0])
    tex_std = tf.constant(param_mean_std['Tex_Para_std'] + 0.000000000000000001, dtype=tf.float32)
    tex_std = tf.transpose(tex_std, [1, 0])

    image_size = 1.0 * image_size
    del param_mean_std

    def unnormalize_labels(batch_size, roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para):
        # roi: shape=(None, 4) or None
        # Shape_Para: shape=(None, 199)
        # Pose_Para: shape=(None, 7)
        # Exp_Para: shape=(None, 29)
        # Color_Para: shape=(None, 6): remove last value as it's always 1
        # Illum_Para: shape=(None, 9): remove last value as it's always 2
        # pt2d: shape=(None, 2, 68)
        # Tex_Para: shape=(None, 40)

        if roi is not None:
            roi *= image_size

        # unnormalize pose param
        pose_para = pose_para * pose_std + pose_avg
        pose_para_new = tf.concat([pose_para[:, 0:3], pose_para[:, 3:] * image_size], axis=1)

        if landmarks is not None:
            landmarks = tf.reshape(landmarks, (batch_size, 2, -1)) * image_size  # (None, 2, 68)

        # unnormalize shape, exp and tex
        shape_para = shape_para * shape_std + shape_avg
        tex_para = tex_para * tex_std + tex_avg
        exp_para = exp_para * exp_std + exp_avg

        # Color_Para: add last value as it's always 1
        # Illum_Para: add last value as it's always 20
        illum_para = tf.concat([illum_para * illum_std + illum_avg, tf.constant(20.0, shape=(batch_size, 1), dtype=tf.float32)], axis=1)
        color_para = tf.concat([color_para * color_std + color_avg, tf.constant(1.0, shape=(batch_size, 1), dtype=tf.float32)], axis=1)

        return roi, landmarks, pose_para_new, shape_para, exp_para, color_para, illum_para, tex_para

    return unnormalize_labels


# def unnormalize_labels(bfm, batch_size, image_size, roi, landmarks, pose_para, shape_para, exp_para, color_para,
#                        illum_para, tex_para):
#     # roi: shape=(None, 4) or None
#     # Shape_Para: shape=(None, 199)
#     # Pose_Para: shape=(None, 7)
#     # Exp_Para: shape=(None, 29)
#     # Color_Para: shape=(None, 6): remove last value as it's always 1
#     # Illum_Para: shape=(None, 9): remove last value as it's always 2
#     # pt2d: shape=(None, 2, 68)
#     # Tex_Para: shape=(None, 40)
#     # image_size
#     # n_tex_para: number of texture coefficients used
#
#     if roi is not None:
#         roi *= image_size
#
#     # translation and scaling
#     pose_para_new = tf.concat([pose_para[:, 0:3], pose_para[:, 3:6] * image_size, pose_para[:, 6:] * image_size / 100000.], axis=1)
#
#     if landmarks is not None:
#         landmarks = tf.reshape(landmarks, (batch_size, 2, -1)) * image_size  # (None, 2, 68)
#
#     # rescale shape, exp and tex by their eigenvalue
#     shape_para *= tf.squeeze(bfm.shape_ev)
#
#     # rescale tex
#     tex_para *= tf.squeeze(bfm.tex_ev)
#
#     # Color_Para: add last value as it's always 1
#     # Illum_Para: add last value as it's always 20
#     color_para = tf.concat([color_para, tf.constant(1.0, shape=(batch_size, 1))], axis=1)
#     illum_para = tf.concat([illum_para, tf.constant(20.0, shape=(batch_size, 1))], axis=1)
#
#     return roi, landmarks, pose_para_new, shape_para, exp_para, color_para, illum_para, tex_para


def fn_extract_coarse_80k(img_filename):
    # label file has the same name as image
    # txt format
    # txt_filename =
    pass


def load_image_from_file(img_file, image_size, resolution, image_format='RGB'):
    """
    load image from file,
    if resolution != image_size, rescale image to resolution
    :param img_file: image file path
    :param image_size: original image size
    :param resolution: target image size
    :return:
    """
    # load image
    img = np.asarray(PIL.Image.open(img_file))
    assert img.shape == (image_size, image_size, 3)
    img = PIL.Image.fromarray(img, image_format)
    # resize image
    if resolution != image_size:
        img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
    return np.array(img)
