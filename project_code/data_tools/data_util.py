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
        lm = mat_data['pt2d']
    elif data_name == 'AFLW_2000':
        lm = mat_data['pt3d_68'][0:2, :]
    else:
        raise Exception('data_name not supported: {0}; only 300W_LP and AFLW_2000 supported'.format(data_name))

    # normalize data
    sp = np.divide(np.subtract(sp, bfm.shape_mu.numpy()), bfm.shape_std.numpy())
    ep = np.divide(np.subtract(ep, bfm.exp_mu.numpy()), bfm.exp_std.numpy())
    tp = np.divide(np.subtract(tp, bfm.tex_mu.numpy()), bfm.tex_std.numpy())
    cp = np.divide(np.subtract(cp, bfm.color_mu.numpy()), bfm.color_std.numpy())
    ip = np.divide(np.subtract(ip, bfm.illum_mu.numpy()), bfm.illum_std.numpy())
    pp = np.divide(np.subtract(pp, bfm.pose_mu.numpy()), bfm.pose_std.numpy())

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


def recover_3dmm_params(image, shape_param, pose_param, exp_param, color_param,
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
    shape_param = np.reshape(shape_param, (-1, 1))
    pose_param = np.reshape(pose_param, (1, -1))
    pose_param[0, 3:] = pose_param[0, 3:] * scale
    exp_param = np.reshape(exp_param, (-1, 1))
    color_param = np.reshape(color_param, (1, -1))
    illum_param = np.reshape(illum_param, (1, -1))
    landmarks = np.reshape(landmarks, (2, -1))
    landmarks = landmarks * scale
    tex_param = np.reshape(tex_param, (-1, 1))

    shape_param = np.add(np.multiply(np.array(shape_param), params_mean_var['Shape_Para_var']),
                         params_mean_var['Shape_Para_mean'])
    pose_param = np.add(np.multiply(np.array(pose_param), params_mean_var['Pose_Para_var']),
                        params_mean_var['Pose_Para_mean'])
    exp_param = np.add(np.multiply(np.array(exp_param), params_mean_var['Exp_Para_var']),
                       params_mean_var['Exp_Para_mean'])
    color_param = np.add(np.multiply(np.array(color_param), params_mean_var['Color_Para_var']),
                         params_mean_var['Color_Para_mean'])
    illum_param = np.add(np.multiply(np.array(illum_param), params_mean_var['Illum_Para_var']),
                         params_mean_var['Illum_Para_mean'])
    tex_param = np.add(np.multiply(np.array(tex_param), params_mean_var['Tex_Para_var']),
                       params_mean_var['Tex_Para_mean'])

    return np.array(image), shape_param, pose_param, exp_param, color_param, illum_param, landmarks, tex_param
