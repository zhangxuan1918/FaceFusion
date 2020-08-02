import os

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import fn_unnormalize_ffhq_labels, split_ffhq_labels
from project_code.face_detector.face_aligner import align_face
from project_code.misc.image_utils import process_reals_supervised


def load_model(pd_model_path):
    return tf.keras.models.load_model(pd_model_path)


def load_images(image_folder, image_names, resolution, detect_face=False):
    for img_name in image_names:
        image_file = os.path.join(image_folder, img_name)

        if detect_face:
            try:
                img = align_face(image_file, output_size=resolution)
                yield img_name, np.array(img)
            except:
                continue
        else:
            img = np.asarray(PIL.Image.open(image_file))
            img = PIL.Image.fromarray(img, 'RGB')
            # resize image
            img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
            img = np.asarray(img)
            yield img_name, img


def add_landmarks(img_rgb, landmarks):
    for j in range(0, 17):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 16:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(17, 22):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 21:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(22, 27):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 26:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(27, 31):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 30:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(31, 36):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 35:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(36, 42):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 41:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(42, 48):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 47:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(48, 60):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 59:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    for j in range(60, 68):
        cv2.circle(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), color=(255, 255, 255), radius=3)
        if j == 67:
            break
        cv2.line(img_rgb, (landmarks[:, j][0], landmarks[:, j][1]), (landmarks[:, j + 1][0], landmarks[:, j + 1][1]),
                 color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    return img_rgb


def inference_and_render_images(images, images_names, model, bfm, unnormalize_labels, rendered_filename_tmp):
    batch_size = len(images)
    reals = tf.convert_to_tensor(images, dtype=tf.uint8)
    reals = process_reals_supervised(x=reals, mirror_augment=False, drange_data=[0, 255], drange_net=[-1, 1])
    est_params = model(reals)
    pose_para, shape_para, exp_para, color_para, illum_para, tex_para = split_ffhq_labels(est_params)
    pose_para, shape_para, exp_para, color_para, illum_para, tex_para = unnormalize_labels(
        batch_size, pose_para, shape_para, exp_para, color_para, illum_para, tex_para)

    # add 0 to t3d z axis
    # 80k dataset only have x, y translation
    pose_para = tf.concat(
        [pose_para[:, :-1], tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float32), pose_para[:, -1:]],
        axis=1)

    landmarks = bfm.get_landmarks(shape_para, exp_para, pose_para, batch_size, image_size, is_2d=True,
                                  is_plot=True)
    image_rendered = render_batch(
        pose_param=pose_para,
        shape_param=shape_para,
        exp_param=exp_para,
        tex_param=tex_para,
        color_param=color_para,
        illum_param=illum_para,
        frame_height=image_size,
        frame_width=image_size,
        tf_bfm=bfm,
        batch_size=batch_size
    ).numpy().astype(np.uint8)

    for i in range(batch_size):
        # input image
        pic_name = '.'.join(images_names[i].split('.')[:-1])

        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        img_rgb = add_landmarks(img_rgb, landmarks[i])

        img_rendered_rgb = cv2.cvtColor(image_rendered[i], cv2.COLOR_BGR2RGB)
        img_rendered_rgb = add_landmarks(img_rendered_rgb, landmarks[i])

        img_all = np.concatenate((img_rgb, img_rendered_rgb), axis=1)
        cv2.imwrite(rendered_filename_tmp.format(pic_name), img_all)


def inverse_rendering(bfm_path, exp_path, pd_model_path, param_mean_std_path, save_to, image_folder, image_names,
                      batch_size, resolution,
                      n_tex_para, n_shape_para):
    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        exp_path=exp_path,
        n_shape_para=n_shape_para,
        n_tex_para=n_tex_para
    )
    model = load_model(pd_model_path=pd_model_path)
    rendered_filename_tmp = os.path.join(save_to, '{0}_rendered.jpg')
    unnormalize_labels = fn_unnormalize_ffhq_labels(param_mean_std_path=param_mean_std_path, image_size=image_size)
    images = []
    images_names = []

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    for img_name, img in load_images(image_folder=image_folder, image_names=image_names, resolution=resolution):
        images.append(img)
        images_names.append(img_name)
        if len(images) == batch_size:
            inference_and_render_images(
                images=images,
                images_names=images_names,
                model=model,
                bfm=bfm,
                unnormalize_labels=unnormalize_labels,
                rendered_filename_tmp=rendered_filename_tmp)

            images = []
            images_names = []

    if images:
        inference_and_render_images(
            images=images,
            images_names=images_names,
            model=model,
            bfm=bfm,
            unnormalize_labels=unnormalize_labels,
            rendered_filename_tmp=rendered_filename_tmp)


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
    n_shape_para = 100
    param_mean_std_path = '/opt/data/face-fuse/stats_80k.npz'
    bfm_path = '/opt/data/BFM/BFM.mat'
    exp_path = '/opt/data/face-fuse/exp_80k.npz'
    pd_model_path = '/opt/data/face-fuse/model/20200725/supervised-exported/'
    image_size = 224

    image_folder = '/opt/project/input/images/ffhq/'
    image_names = ['69174.png', '69533.png', '69282.png', '07377.png', '10154.png', '20983.png', '22976.png',
                   '34895.png',
                   '07248.png', '07248.png', '68241.png', '46096.png']

    save_rendered_to = '/opt/project/data/output/ffhq'

    inverse_rendering(
        bfm_path, exp_path, pd_model_path, param_mean_std_path, save_rendered_to, image_folder=image_folder,
        image_names=image_names, batch_size=2, resolution=image_size, n_tex_para=n_tex_para, n_shape_para=n_shape_para
    )
