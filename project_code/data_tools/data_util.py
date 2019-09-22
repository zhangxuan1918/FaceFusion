import numpy as np
import scipy.io as sio
import tensorflow as tf

from data_tools.data_const import face_vgg2_input_mean
from morphable_model.model.morphable_model import FFTfMorphableModel


def load_image_3dmm(im_size_pre_shift: int, im_size: int, tx: int, ty: int, im_file: str):
    """
    load and preprocess images for 3dmm
    original image size (450, 450)

    * resize image to (im_size_pre_shift, im_size_pre_shift)
    * shift image by (tx, ty) and resize image to (im_size, im_size)
    :param im_file:
    :param im_size_pre_shift: image size before we translate
    :param im_size: target image size
    :param tx: x translation
    :param ty: y translation
    :return:
    """

    image = tf.io.read_file(im_file)
    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.image.resize(image, (im_size_pre_shift, im_size_pre_shift))
    # translate and resize image to (224, 224)
    image = tf.image.crop_to_bounding_box(image, tx, ty, im_size, im_size)
    # normalize image to [-1, 1]
    image = (image / 127.5) - 1
    # image -= face_vgg2_input_mean
    return image


def load_mat_3dmm(bfm: FFTfMorphableModel, data_name: str, mat_file: str, im_size_pre_shift: int, tx: int, ty: int):
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
        # Tex_Para: shape=(40, 1)
    AFLW_200 data set
        # Shape_Para: shape=(199, 1)
        # Pose_Para: shape=(1, 7)
        # Exp_Para: shape=(29, 1)
        # Color_Para: shape=(1, 7)
        # Illum_Para: shape=(1, 10)
        # pt3d_68: shape=(3, 68)
        # Tex_Para: shape=(40, 1)
    total params: 587
    :param: bfm: face morphable model
    :param: data_name
    :param: mat_file
    :param: tx: x translation
    :param: ty: y translation
    :param: image_size: target image size
    :param im_size_pre_shift: image size before we translate
    :return:
    """

    def _read_mat(mat_file, tx, ty):
        mat_data = sio.loadmat(mat_file.numpy())

        sp = mat_data['Shape_Para']
        ep = mat_data['Exp_Para']
        tp = mat_data['Tex_Para'][:40, :]
        cp = mat_data['Color_Para']
        ip = mat_data['Illum_Para']
        pp = mat_data['Pose_Para']

        # get landmark
        if data_name == '300W_LP':
            lm = mat_data['pt2d']
        elif data_name == 'AFLW_2000':
            lm = mat_data['pt3d_68'][0:2, :]
        else:
            raise Exception('data_name not supported: {0}; only 300W_LP and AFLW_2000 supported'.format(data_name))

        # normalize data
        sp = np.divide(np.subtract(sp, bfm.stats_shape_mu.numpy()), bfm.stats_shape_std.numpy())
        ep = np.divide(np.subtract(ep, bfm.stats_exp_mu.numpy()), bfm.stats_exp_std.numpy())
        tp = np.divide(np.subtract(tp, bfm.stats_tex_mu.numpy()), bfm.stats_tex_std.numpy())
        cp = np.divide(np.subtract(cp, bfm.stats_color_mu.numpy()), bfm.stats_color_std.numpy())
        ip = np.divide(np.subtract(ip, bfm.stats_illum_mu.numpy()), bfm.stats_illum_std.numpy())

        # update pose param with random translation
        pp[0, 3:5] = im_size_pre_shift / 450.
        pp[0, 6] = im_size_pre_shift / 450.
        pp[0, 3], pp[0, 4] = pp[0, 3] - ty, pp[0, 4] - (32. - tx)
        pp = np.divide(np.subtract(pp, bfm.stats_pose_mu.numpy()), bfm.stats_pose_std.numpy())

        # update landmark param with random translation
        lm *= im_size_pre_shift / 450.
        lm[0], lm[1] = lm[0] - ty, lm[1] - tx
        return sp, pp, ep, cp, ip, tp, lm

    sp, pp, ep, cp, ip, tp, lm = tf.py_function(_read_mat, [mat_file, 1.0 * tx, 1.0 * ty], [tf.float32] * 7)
    return {
                'shape': sp,
                'pose': pp,
                'exp': ep,
                'color': cp,
                'illum': ip,
                'tex': tp,
                'landmark': lm
            }


def load_3dmm_data(
        bfm: FFTfMorphableModel,
        im_size_pre_shift: int,
        im_size: int,
        data_name: str,
        image_file: str,
        mat_file: str):
    tx = np.random.randint(0, 33, dtype=np.int32)
    ty = np.random.randint(0, 33, dtype=np.int32)
    image = load_image_3dmm(im_file=image_file, im_size_pre_shift=im_size_pre_shift, im_size=im_size, tx=tx, ty=ty)
    mat_dict = load_mat_3dmm(bfm=bfm, data_name=data_name, mat_file=mat_file, im_size_pre_shift=im_size_pre_shift, tx=tx, ty=ty)
    return image, mat_dict
