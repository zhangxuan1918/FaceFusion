import imageio
import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
import matplotlib.pyplot as plt
from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, fn_unnormalize_300W_LP_labels
from project_code.training import dataset


def display(tfrecord_dir, bfm_path, param_mean_std_path, image_size, num_images=5, n_tex_para=40):
    print('Loading sdataset %s' % tfrecord_dir)

    batch_size = 4
    dset = dataset.TFRecordDatasetSupervised(
        tfrecord_dir=tfrecord_dir, batch_size=batch_size, repeat=False, shuffle_mb=0)
    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        n_tex_para=n_tex_para
    )

    idx = 0
    filename = '/opt/project/output/verify_dataset/supervised/20200525/image_batch_{0}_indx_{1}.jpg'
    unnormalize_labels = fn_unnormalize_300W_LP_labels(param_mean_std_path=param_mean_std_path, image_size=image_size)
    while idx < num_images:
        try:
            image_tensor, labels_tensor = dset.get_minibatch_tf()
        except tf.errors.OutOfRangeError:
            break

        # render images using labels
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = split_300W_LP_labels(labels_tensor)
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = unnormalize_labels(batch_size, roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para)
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
            ax.imshow(image_tensor[i].numpy().astype(np.uint8))
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

            plt.savefig(filename.format(idx, i))

        idx += 1

    print('\nDisplayed %d images' % idx)

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tfrecord_dir = '/opt/data/face-fuse/supervised/train/'
    param_mean_std_path = '/opt/data/face-fuse/stats_300W_LP.npz'
    bfm_path = '/opt/data/BFM/BFM.mat'
    image_size = 224
    num_images = 8
    display(tfrecord_dir, bfm_path, param_mean_std_path, image_size, num_images=num_images)