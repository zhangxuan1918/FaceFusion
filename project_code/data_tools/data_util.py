import numpy as np
import scipy.io as sio
import tensorflow as tf

from project_code.data_tools.data_const import keys_3dmm_params

params_mean_var = np.load('/opt/project/project_code/data/3dmm/300W_LP_mean_var/stats_300W_LP.npz')


def load_image_3dmm(image_file, output_size=224):
    """
    load and preprocess images for 3dmm
    original image size (450, 450)
    we need to scale down image to (224, 224)
    :param image_file:
    :return:
    """

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)

    # resize image from (450, 450) to (224, 224)
    image = tf.image.resize(image, [output_size, output_size])
    # normalize image to [-1, 1]
    image = (image / 127.5) - 1
    return image


def load_labels_3dmm(label_file, image_original_size=450, image_rescaled_size=224):
    """
    load labels for image
    we rescale image from size (450, 450) to (224, 224)
    we need to adapt the 3dmm params

    # Shape_Para: shape=(199, 1) => (199,)
    # Pose_Para: shape=(1, 7) => (7,)
    # Exp_Para: shape=(29, 1) => (29,)
    # Color_Para: shape=(1, 7) => (7,)
    # Illum_Para: shape=(1, 10) => (10,)
    # pt2d: shape=(2, 68) => (2, 68)
        flatten to (136, )
    # Tex_Para: shape=(199, 1) => (199,)

    total params: 587
    :param: label_file:
    :param: image_original_size: 450
    :param: image_rescaled_size: 224
    :return: label of shape (587, )
    """

    def _read_mat(label_file):
        mat_contents = sio.loadmat(label_file.numpy())

        labels = []
        scale_ratio = 1.0 * image_rescaled_size / image_original_size

        for key in keys_3dmm_params:
            value = mat_contents[key]
            if key == 'pt2d':
                # for landmark we just normalize by image size
                value = np.reshape(value, (-1, 1))
                value *= scale_ratio
            elif key == 'Pose_param':
                # mean
                value_mean = params_mean_var[key + '_mean']
                # std
                value_var = params_mean_var[key + '_var']
                # rescale pose param as the image is rescaled
                value[0, 3:] = value[0, 3:] * scale_ratio
                value_mean[0, 3:] = value_mean[0, 3:] * scale_ratio
                value_var[0, 3:] = value_var[0, 3:] * scale_ratio
                value = np.divide(np.subtract(value, value_mean), value_var)
            else:
                # mean
                value_mean = params_mean_var[key + '_mean']
                # std
                value_var = params_mean_var[key + '_var']
                value = np.divide(np.subtract(value, value_mean), value_var)
            labels.append(tf.squeeze(tf.convert_to_tensor(value, dtype=tf.float32)))

        labels_flatten = tf.concat(labels, axis=0)
        return labels_flatten

    labels = tf.py_function(_read_mat, [label_file], tf.float32)
    labels.set_shape((587,))
    return labels


def load_image_labels_3dmm(image_file, label_file):
    return load_image_3dmm(image_file=image_file), load_labels_3dmm(label_file=label_file)


def recover_3dmm_params(shape_param, pose_param, exp_param, color_param,
                        illum_param, tex_param, landmarks, output_size, input_size):
    """
    add mean and times std
    reshape the params into right shape

    :param: image: original image
    :param: shape_param: shape=(199,) => (199, 1)
    :param: pose_param: shape=(7,) => (1, 7)
    :param: exp_param: shape=(29,) => (29, 1)
    :param: color_param: shape=(7,) => (1, 7)
    :param: illum_param: shape=(10,) => (1, 10)
    :param: landmarks: shape=(2, 68) => (2, 68)
    :param: tex_param: shape=(199,) => (199, 1)
    :returns:
        shape_para: (199, 1)
        pose_para: (1, 7)
        exp_para: (29, 1)
        color_para: (1, 7)
        illum_para: (1, 10)
        landmarks: (2, 68)
        tex_para: (199, 1)
    """
    # reshape
    scale = 1.0 * output_size / input_size
    shape_param = tf.expand_dims(shape_param, axis=1)
    pose_param = tf.expand_dims(pose_param, axis=0)
    pose_param = tf.concat(pose_param[0, 0:])
    exp_param = tf.expand_dims(exp_param, axis=1)
    color_param = tf.expand_dims(color_param, axis=0)
    illum_param = tf.expand_dims(illum_param, axis=0)
    landmarks = landmarks * scale
    tex_param = tf.expand_dims(tex_param, axis=1)


    shape_param = shape_param * params_mean_var['Shape_Para_var'] + params_mean_var['Shape_Para_mean']
    pose_param = pose_param * params_mean_var['Pose_Para_var'] + params_mean_var['Pose_Para_mean']
    exp_param = exp_param * params_mean_var['Exp_Para_var'] + params_mean_var['Exp_Para_mean']
    color_param = color_param + params_mean_var['Color_Para_var'] + params_mean_var['Color_Para_mean']
    illum_param = illum_param + params_mean_var['Illum_Para_var'] + params_mean_var['Illum_Para_mean']
    tex_param = tex_param + params_mean_var['Tex_Para_var'] + params_mean_var['Tex_Para_mean']

    return shape_param, pose_param, exp_param, color_param, illum_param, landmarks, tex_param, scale
