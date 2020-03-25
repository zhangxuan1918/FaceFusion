import os
import tensorflow_addons as tfa
import imageio
import numpy as np
from project_code.create_tfrecord.export_tfrecord_util import load_image_from_file
import tensorflow as tf

from project_code.misc.image_utils import process_reals_unsupervised


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
    masks_tensor = tf.convert_to_tensor(masks, dtype=tf.float32)
    _, images_tensor, masks_tensor = process_reals_unsupervised(images_tensor, masks_tensor, batch_size=len(images),
                                                                mirror_augment=False,
                                                                drange_data=[0, 255], drange_net=[-1, 1],
                                                                resolution=224,
                                                                max_rotate_degree=30)

    for image, mask, img_name in zip(images_tensor, masks_tensor, image_names):
        to_save_file = os.path.join(to_save, img_name)
        image_mask = np.concatenate((image.numpy().astype(np.uint8), mask.numpy().astype(np.uint8)), axis=0)
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
        resolution=256
    )
