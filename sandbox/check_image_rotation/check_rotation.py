import os
import tensorflow_addons as tfa
import imageio
import numpy as np
from project_code.create_tfrecord.export_tfrecord_util import load_image_from_file
import tensorflow as tf


def rotate(image_names, image_folder, to_save, image_size=256, resolution=256):
    images = []
    masks = []
    for img_name in image_names:
        img_file = os.path.join(image_folder, img_name)
        img = load_image_from_file(img_file=img_file, image_size=image_size,
                                   resolution=resolution, image_format='RGB')
        images.append(img)

        msk_file = os.path.join(mask_folder, img_name)
        mask = load_image_from_file(img_file=msk_file, image_size=image_size,
                                    resolution=resolution, image_format='RGB')
        masks.append(mask)

    images_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    rad = 30. * np.pi / 180.
    angles = tf.random.uniform((tf.shape(images_tensor)[0],), maxval=rad, minval=-rad)
    images_rotated = tfa.image.rotate(
        images_tensor,
        angles,
        interpolation='NEAREST',
        name=None
    ).numpy().astype(np.uint8)

    masks_tensor = tf.convert_to_tensor(masks, dtype=tf.float32)
    masks_rotated = tfa.image.rotate(
        masks_tensor,
        angles,
        interpolation='NEAREST',
        name=None
    ).numpy().astype(np.uint8)

    for image, mask, img_name in zip(images_rotated, masks_rotated, image_names):
        to_save_file = os.path.join(to_save, img_name)
        image_mask = np.concatenate((image.astype(np.uint8), mask.astype(np.uint8)), axis=0)
        imageio.imsave(to_save_file, image_mask)


if __name__ == '__main__':
    image_names = ['AFW_2805422179_3_4.png', 'AFW_4492032921_1_6.png', 'AFW_4492032921_1_17.png']
    dataset = 'AFW'
    image_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/image/{0}'.format(dataset)
    mask_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/mask_img/{0}'.format(dataset)
    to_save = './output'

    rotate(
        image_names=image_names,
        image_folder=image_folder,
        to_save=to_save,
        image_size=256,
        resolution=224
    )
