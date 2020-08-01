import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, fn_unnormalize_300W_LP_labels, \
    fn_unnormalize_80k_labels, split_80k_labels, fn_unnormalize_ffhq_labels, split_ffhq_labels
from project_code.misc.image_utils import process_reals_unsupervised, process_reals_supervised


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
        yield img_name, img


def check_prediction_adhoc(bfm_path, exp_path, pd_model_path, param_mean_std_path, save_to, image_folder, image_names,
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
    filename = os.path.join(save_to, '{0}.jpg')
    unnormalize_labels = fn_unnormalize_ffhq_labels(param_mean_std_path=param_mean_std_path, image_size=image_size)
    images = []
    images_names = []
    for img_name, img in load_images(image_folder=image_folder, image_names=image_names, resolution=resolution):
        images.append(img)
        images_names.append(img_name)
        if len(images) == batch_size:
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
                fig = plt.figure()
                # input image
                ax = fig.add_subplot(1, 2, 1)
                ax.imshow(images[i])
                ax.plot(landmarks[i, 0, 0:17], landmarks[i, 1, 0:17], marker='o', markersize=2, linestyle='-',
                        color='w', lw=2)
                ax.plot(landmarks[i, 0, 17:22], landmarks[i, 1, 17:22], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 22:27], landmarks[i, 1, 22:27], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 27:31], landmarks[i, 1, 27:31], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 31:36], landmarks[i, 1, 31:36], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 36:42], landmarks[i, 1, 36:42], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 42:48], landmarks[i, 1, 42:48], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 48:60], landmarks[i, 1, 48:60], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)
                ax.plot(landmarks[i, 0, 60:68], landmarks[i, 1, 60:68], marker='o', markersize=2,
                        linestyle='-', color='w', lw=2)

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(image_rendered[i])
                ax2.plot(landmarks[i, 0, 0:17], landmarks[i, 1, 0:17], marker='o', markersize=2, linestyle='-',
                         color='w', lw=2)
                ax2.plot(landmarks[i, 0, 17:22], landmarks[i, 1, 17:22], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 22:27], landmarks[i, 1, 22:27], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 27:31], landmarks[i, 1, 27:31], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 31:36], landmarks[i, 1, 31:36], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 36:42], landmarks[i, 1, 36:42], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 42:48], landmarks[i, 1, 42:48], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 48:60], landmarks[i, 1, 48:60], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)
                ax2.plot(landmarks[i, 0, 60:68], landmarks[i, 1, 60:68], marker='o', markersize=2,
                         linestyle='-', color='w', lw=2)

                plt.savefig(filename.format(images_names[i]))
            images = []
            images_names = []


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
    save_rendered_to = '/opt/project/output/adhoc_predict/supervised_ffhq/random/'
    bfm_path = '/opt/data/BFM/BFM.mat'
    exp_path = '/opt/data/face-fuse/exp_80k.npz'
    pd_model_path = '/opt/data/face-fuse/model/20200725/supervised-exported/'
    image_size = 224

    # image_folder = '/opt/project/input/images/ffhq/'
    # image_names = ['69174.png', '69533.png', '68241.png', '69282.png']
    # image_names = ['07248.png', '07377.png', '10154.png', '20658.png', '20983.png', '22976.png', '27516.png',
    #                '34895.png', '46096.png', '46955.png', '52354.png', '52675.png', '56202.png', '58457.png',
    #                '64807.png', '65716.png']

    # image_names = ['09711.png', '19426.png', '29141.png', '38856.png', '48571.png', '58286.png', '09712.png',
    #                '19427.png', '29142.png', '38857.png', '48572.png', '58287.png', '09713.png', '19428.png',
    #                '29143.png', '38858.png', '48573.png', '58288.png', '09714.png', '19429.png', '29144.png',
    #                '38859.png', '48574.png', '58289.png']

    # image_folder = '/opt/data/300W_LP/AFW/'
    # image_names = ['AFW_2805422179_3_17.jpg', 'AFW_4492032921_1_2.jpg', 'AFW_955659370_2_6.jpg',
    #                'AFW_2805422179_3_1.jpg', 'AFW_4492032921_1_3.jpg', 'AFW_955659370_2_7.jpg',
    #                'AFW_2805422179_3_2.jpg', 'AFW_4492032921_1_4.jpg', 'AFW_955659370_2_8.jpg',
    #                'AFW_2805422179_3_3.jpg', 'AFW_4492032921_1_5.jpg', 'AFW_955659370_2_9.jpg',
    #                'AFW_2805422179_3_4.jpg', 'AFW_4492032921_1_6.jpg']

    # image_folder = '/opt/data/300W_LP/HELEN/'
    # image_names = ['HELEN_2442107375_1_3.jpg',
    #                'HELEN_2422195369_1_1.jpg',
    #                'HELEN_3223115234_1_3.jpg',
    #                'HELEN_2422195369_1_11.jpg',
    #                'HELEN_3223115234_1_11.jpg',
    #                'HELEN_2421887901_1_7.jpg',
    #                'HELEN_3222262589_1_3.jpg',
    #                'HELEN_2421887901_1_1.jpg',
    #                'HELEN_3222262589_1_15.jpg',
    #                'HELEN_2421887901_1_11.jpg',
    #                'HELEN_3222261569_1_7.jpg',
    #                'HELEN_2421145346_1_3.jpg',
    #                'HELEN_3222261569_1_13.jpg',
    #                'HELEN_2421145346_1_14.jpg',
    #                'HELEN_3220402975_1_9.jpg',
    #                'HELEN_2421145346_1_0.jpg']

    image_folder = '/opt/project/input/images/random/'
    image_names = ['pic2.jpeg', 'pic2.jpeg', 'pic3.jpeg', 'pic4.jpeg', 'pic5.jpeg', 'pic6.jpeg', 'pic7.jpeg',
                   'pic8.jpeg', 'pic9.jpeg', 'pic10.jpeg', 'pic11.jpeg', 'pic12.jpeg', 'pic13.jpeg', 'pic14.jpeg',
                   'pic15.jpeg', 'pic16.jpeg', 'pic17.jpeg', 'pic18.jpeg'
                   ]

    # image_folder = '/opt/data/300W_LP/IBUG/'
    # image_names = ['IBUG_image_99_9.jpg',
    #                'IBUG_image_069_1_4.jpg',
    #                'IBUG_image_119_8.jpg',
    #                'IBUG_image_068_1_4.jpg',
    #                'IBUG_image_067_1_5.jpg',
    #                'IBUG_image_118_7.jpg',
    #                'IBUG_image_066_1_8.jpg',
    #                'IBUG_image_115_1.jpg',
    #                'IBUG_image_061_1_9.jpg',
    #                'IBUG_image_059_7.jpg',
    #                'IBUG_image_114_5.jpg',
    #                'IBUG_image_113_3.jpg',
    #                'IBUG_image_057_1_1.jpg',
    #                'IBUG_image_111_8.jpg',
    #                'IBUG_image_054_10.jpg',
    #                'IBUG_image_053_6.jpg']

    check_prediction_adhoc(
        bfm_path, exp_path, pd_model_path, param_mean_std_path, save_rendered_to, image_folder=image_folder,
        image_names=image_names, batch_size=2, resolution=image_size, n_tex_para=n_tex_para, n_shape_para=n_shape_para
    )
