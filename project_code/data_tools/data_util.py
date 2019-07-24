import numpy as np
import scipy.io as sio
import tensorflow as tf

from morphable_model.model.morphable_model import FFTfMorphableModel


def load_image_3dmm(image_file: str):
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
    image = tf.image.resize(image, [224, 224])
    # normalize image to [-1, 1]
    image = (image / 127.5) - 1
    return image


def load_mat_3dmm(bfm: FFTfMorphableModel, data_name: str, mat_file: str):
    """
    load labels for image
    we rescale image from size (450, 450) to (224, 224)
    we need to adapt the 3dmm params
    300W_LP data set
        # Shape_Para: shape=(199, 1)
        # Pose_Para: shape=(1, 7)
        # Exp_Para: shape=(29, 1)
        # Color_Para: shape=(1, 7)
        # Illum_Para: shape=(1, 10)
        # pt2d: shape=(2, 68)
        # Tex_Para: shape=(199, 1)
    AFLW_200 data set
        # Shape_Para: shape=(199, 1)
        # Pose_Para: shape=(1, 7)
        # Exp_Para: shape=(29, 1)
        # Color_Para: shape=(1, 7)
        # Illum_Para: shape=(1, 10)
        # pt3d_68: shape=(3, 68)
        # Tex_Para: shape=(199, 1)
    total params: 587
    :param: data_name
    :param: label_file:
    :param: image_original_size: 450
    :param: image_rescaled_size: 224
    :return:
    """
    mat_data = sio.loadmat(mat_file)

    sp = mat_data['Shape_Para']
    ep = mat_data['Exp_Para']
    tp = mat_data['Tex_Para']
    cp = mat_data['Color_Para']
    ip = mat_data['Illum_Para']
    pp = mat_data['Pose_Para']

    if data_name == '300W_LP':
        lm = mat_data['pt2d'] * 224 / 450
    elif data_name == 'AFLW_2000':
        lm = mat_data['pt3d_68'][0:2, :] * 224 / 450
    else:
        raise Exception('data_name not supported: {0}; only 300W_LP and AFLW_2000 supported'.format(data_name))

    # normalize data
    sp = np.divide(np.subtract(sp, bfm.stats_shape_mu.numpy()), bfm.stats_shape_std.numpy())
    ep = np.divide(np.subtract(ep, bfm.stats_exp_mu.numpy()), bfm.stats_exp_std.numpy())
    tp = np.divide(np.subtract(tp, bfm.stats_tex_mu.numpy()), bfm.stats_tex_std.numpy())
    cp = np.divide(np.subtract(cp, bfm.stats_color_mu.numpy()), bfm.stats_color_std.numpy())
    ip = np.divide(np.subtract(ip, bfm.stats_illum_mu.numpy()), bfm.stats_illum_std.numpy())
    pp[0, 3:] = pp[0, 3:] * 224 / 450
    pp = np.divide(np.subtract(pp, bfm.stats_pose_mu.numpy()), bfm.stats_pose_std.numpy())

    return \
        {
            'shape': sp,
            'pose': pp,
            'exp': ep,
            'color': cp,
            'illum': ip,
            'tex': tp,
            'landmark': lm
        }


def load_3dmm_data_gen(
        bfm: FFTfMorphableModel,
        data_name: str,
        image_file: str,
        mat_file: str):
    for im_file, mat_file in zip(image_file, mat_file):
        image = load_image_3dmm(image_file=im_file)
        mat_dict = load_mat_3dmm(bfm=bfm, data_name=data_name, mat_file=mat_file)

        yield image, mat_dict
