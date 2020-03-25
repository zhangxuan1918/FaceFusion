import os
from PIL import Image
import imageio
import numpy as np


def compute_masked_face(image_name, image_folder, mask_folder, to_save_folder):

    image_file = os.path.join(image_folder, image_name)
    mask_file = os.path.join(mask_folder, image_name)
    image = np.array(Image.open(image_file))
    mask = np.array(Image.open(mask_file))
    image[mask == 0] = 0

    to_save_file = os.path.join(to_save_folder, image_name)
    imageio.imsave(to_save_file, image)


if __name__ == '__main__':
    image_name = 'image00002.png'
    image_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/image/AFLW2000'
    mask_folder = '/opt/data/nonlinear_face_3dmm/300W_LP/mask_img/AFLW2000'
    to_save_folder = '/opt/project/output/masked_image/'

    compute_masked_face(
        image_name=image_name,
        image_folder=image_folder,
        mask_folder=mask_folder,
        to_save_folder=to_save_folder
    )
