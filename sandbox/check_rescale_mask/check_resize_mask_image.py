import os

import imageio
import numpy as np
from project_code.create_tfrecord.export_tfrecord_util import load_image_from_file


def rescale_image_and_mask(image_names, image_folder, mask_folder, to_save, image_size=256, resolution=224):
    for img_name in image_names:
        img_file = os.path.join(image_folder, img_name)
        img = load_image_from_file(img_file=img_file, image_size=image_size,
                                   resolution=resolution, image_format='RGB')

        msk_file = os.path.join(mask_folder, img_name)
        mask = load_image_from_file(img_file=msk_file, image_size=image_size,
                                    resolution=resolution, image_format='RGB')

        msked_image = np.copy(img)
        msked_image[mask == 0] = 0
        image_mask = np.concatenate((img.astype(np.uint8), mask.astype(np.uint8), msked_image.astype(np.uint8)), axis=0)

        to_save_file = os.path.join(to_save, img_name)
        imageio.imsave(to_save_file, image_mask)


if __name__ == '__main__':
    image_names = ['AFW_2805422179_3_4.png', 'AFW_4492032921_1_6.png', 'AFW_4492032921_1_17.png']
    dataset = 'AFW'
    image_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/image/{0}'.format(dataset)
    mask_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/mask_img/{0}'.format(dataset)
    to_save = './output'

    rescale_image_and_mask(
        image_names=image_names,
        image_folder=image_folder,
        mask_folder=mask_folder,
        to_save=to_save,
        image_size=256,
        resolution=224
    )
