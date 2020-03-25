import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def process_reals_supervised(x, mirror_augment, drange_data, drange_net):
    x = tf.cast(x, tf.float32)
    x = adjust_dynamic_range(x, drange_data, drange_net)

    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            s = tf.shape(x)
            mask = tf.random.uniform([s[0], 1, 1, 1], 0.0, 1.0)
            mask = tf.tile(mask, [1, s[1], s[2], s[3]])
            x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))

    return x


def random_crop(images, masks, width, height, batch_size):
    """
    random crop images and mask
    :param images: [batch, height, width]
    :param masks: [batch, height, width]
    :param width: int
    :param height: int
    :return:
    """
    assert tf.shape(images)[1] >= height
    assert tf.shape(images)[1] >= width
    assert tf.shape(images) == tf.shape(masks)

    x = np.random.random_integers(0, tf.shape(images)[2] - width, size=batch_size)
    y = np.random.random_integers(0, tf.shape(images)[1] - height, size=batch_size)
    images = images[:, y:y + height, x:x + width]
    masks = masks[:, y:y + height, x:x + width]
    return images, masks


def process_reals_unsupervised(images, masks, batch_size, mirror_augment, drange_data, drange_net, resolution=224,
                               max_rotate_degree=30):
    images = tf.cast(images, tf.float32)

    # random crop
    with tf.name_scope('CropAugment'):
        images, masks = random_crop(
            images=images,
            masks=masks,
            width=resolution,
            height=resolution,
            batch_size=batch_size
        )
    # random rotation
    with tf.name_scope('RotationAugment'):
        rad = max_rotate_degree * np.pi / 180.
        angles = tf.random.uniform((tf.shape(images)[0],), maxval=rad, minval=-rad)
        images = tfa.image.rotate(images, angles, interpolation='NEAREST')
        masks = tfa.image.rotate(masks, angles, interpolation='NEAREST')

    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            images = tf.image.flip_left_right(images)
            masks = tf.image.flip_left_right(masks)

    images = adjust_dynamic_range(images, drange_data, drange_net)

    tf.debugging.assert_shapes([(images, (batch_size, resolution, resolution)),
                                (masks, (batch_size, resolution, resolution))])

    return images, masks
