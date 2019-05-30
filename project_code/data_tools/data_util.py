import tensorflow as tf
import scipy.io as sio

from project_code.data_tools.data_const import keys_3dmm_params


def load_image_3dmm(image_file, output_size=224):
    """
    load and preprocess images for 3dmm
    original image size (450, 450)
    we need to scale down image to (224, 224)
    :param image_file:
    :return:
    """

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

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
    # Tex_Para: shape=(199, 1) => (199,)
    :param label_file:
    :return:
    """
    mat_contents = sio.loadmat(label_file)

    labels = []
    scale_ratio = 1.0 * image_rescaled_size / image_original_size

    for key in keys_3dmm_params:
        value = mat_contents[key]
        if key == 'Pose_param':
            # rescale pose param as the image is rescaled
            value[0, 3:] = value[0, 3:] * scale_ratio
        labels.append(tf.squeeze(tf.convert_to_tensor(value)))

    return labels


def load_image_labels_3dmm(image_file, label_file):
    return load_image_3dmm(image_file=image_file), load_labels_3dmm(label_file=label_file)