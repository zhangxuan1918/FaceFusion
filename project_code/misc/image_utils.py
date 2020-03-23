import tensorflow as tf
import numpy as np


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def process_reals(x, mirror_augment, drange_data, drange_net, random_crop_augment=False, random_rotation_augment=False,
                  resolution=224, seed=1987):
    x = tf.cast(x, tf.float32)

    if random_crop_augment:
        # random crop
        with tf.name_scope('CropAugment'):
            x = tf.image.random_crop(x, (resolution, resolution, 3), seed=seed)

    if random_rotation_augment:
        with tf.name_scope('RotationAugment'):
            rg = tf.random.uniform(shape=(tf.shape(x)[0], ), minval=-30, maxval=30, seed=seed)
            x = tf.keras.preprocessing.image.random_rotation(
                x, rg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, interpolation_order=1
            )

    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.image.flip_left_right(x)

    ax = adjust_dynamic_range(x, drange_data, drange_net)
    return x, ax
