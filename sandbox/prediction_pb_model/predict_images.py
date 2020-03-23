import os

import PIL
import imageio
import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, unnormalize_labels
from project_code.misc.image_utils import process_reals


def load_model(pd_model_path):
    return tf.keras.models.load_model(pd_model_path)


def load_images(image_folder, image_names, resolution):
    for img_name in image_names:
        image_file = os.path.join(image_folder, img_name)
        img = np.asarray(PIL.Image.open(image_file))
        img = PIL.Image.fromarray(img, 'RGB')
        # resize image
        img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
        img = np.asarray(img)
        yield img


def check_prediction_adhoc(bfm_path, pd_model_path, save_to, image_folder, image_names, batch_size, resolution,
                           n_tex_para):
    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        n_tex_para=n_tex_para
    )
    model = load_model(pd_model_path=pd_model_path)
    filename = os.path.join(save_to, 'image_batch_{0}_indx_{1}.jpg')

    images = []
    idx = 0
    for img in load_images(image_folder=image_folder, image_names=image_names, resolution=resolution):
        images.append(img)

        if len(images) == batch_size:
            reals = tf.convert_to_tensor(images, dtype=tf.uint8)
            reals = process_reals(x=reals, mirror_augment=False, drange_data=[0, 255], drange_net=[-1, 1])
            est_params = model(reals)

            _, _, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_300W_LP_labels(est_params)

            _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = unnormalize_labels(
                bfm, batch_size, resolution, None, None, est_pp, est_shape, est_exp, est_color, est_illum, est_tex)

            est_image = render_batch(
                pose_param=est_pp,
                shape_param=est_shape,
                exp_param=est_exp,
                tex_param=est_tex,
                color_param=est_color,
                illum_param=est_illum,
                frame_height=resolution,
                frame_width=resolution,
                tf_bfm=bfm,
                batch_size=batch_size
            ).numpy().astype(np.uint8)

            for i in range(len(images)):
                images_to_save = np.concatenate((images[i], est_image[i]), axis=0)
                imageio.imsave(filename.format(idx, i), images_to_save)
            idx += 1
            images = []

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

        except RuntimeError as e:
            raise e

    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='/opt/project/examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    save_rendered_to = '/opt/project/output/adhoc_predict/supervised/ffhq'
    bfm_path = '/opt/data/BFM/BFM.mat'
    pd_model_path = '/opt/data/face-fuse/model/20200322/supervised-exported/'
    image_size = 224
    image_folder = '/opt/project/input/images/ffhq/'
    image_names = ['09711.png', '19426.png', '29141.png', '38856.png', '48571.png', '58286.png', '09712.png',
                   '19427.png', '29142.png', '38857.png', '48572.png', '58287.png', '09713.png', '19428.png',
                   '29143.png', '38858.png', '48573.png', '58288.png', '09714.png', '19429.png', '29144.png',
                   '38859.png', '48574.png', '58289.png']
    check_prediction_adhoc(
        bfm_path, pd_model_path, save_rendered_to, image_folder=image_folder,
        image_names=image_names, batch_size=4, resolution=image_size, n_tex_para=n_tex_para
    )
