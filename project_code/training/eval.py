import os

import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_code.data_tools.data_util import recover_3dmm_params


def update_tf_summary(var_name, metric, step):
    tf.summary.scalar(var_name, metric.result(), step=step)
    metric.reset_states()


def save_rendered_image_for_eval(
        image,
        bfm,
        render_image_size,
        input_image_size,
        shape_param,
        pose_param,
        exp_param,
        color_param,
        illum_param,
        tex_param,
        landmark_param,
        landmark_groundtruth,
        save_eval_to_folder,
        epoch,
        batch_id,
        image_id
):
    # plot est
    shape_param, pose_param, exp_param, color_param, illum_param, \
        landmarks, tex_param, resale = recover_3dmm_params(
            shape_param=shape_param,
            pose_param=pose_param,
            exp_param=exp_param,
            color_param=color_param,
            illum_param=illum_param,
            tex_param=tex_param,
            landmarks=landmark_param,
            output_size=render_image_size,
            input_size=input_image_size
    )

    image = tf.image.resize(image, [render_image_size, render_image_size])

    save_to_file = os.path.join(save_eval_to_folder, 'epoch_{epoch}_batch_{batch_id}_image_{image_id}.jpg'.format(
        epoch=epoch, batch_id=batch_id, image_id=image_id
    ))

    image_rendered = render_2(
        angles_grad=pose_param[0, 0:3],
        t3d=pose_param[0, 3:6] * resale,
        scaling=pose_param[0, 6] * resale,
        shape_param=shape_param,
        exp_param=exp_param,
        tex_param=tex_param,
        color_param=color_param,
        illum_param=illum_param,
        frame_height=450,
        frame_width=450,
        tf_bfm=bfm
    )
    save_images(
        images=[image.numpy(), image_rendered.numpy().astype(np.uint8)],
        landmarks=[landmark_groundtruth.numpy(), landmarks.numpy()],
        titles=['orignal', 'rendered'],
        file_to_save=save_to_file
    )


def save_images(images, titles, file_to_save=None, landmarks=None):
    if file_to_save is not None:
        fig = plt.figure()
        n_images = len(images)

        if landmarks is None or len(landmarks) != n_images:
            landmarks = [None] * 2

        for i, image, lm, title in enumerate(zip(images, landmarks, titles)):
            ax = fig.add_subplot(1, n_images, i + 1)
            ax.imshow(image)
            ax.set_title(title)

            if lm is not None:
                ax.plot(lm[0, 0:17], lm[1, 0:17], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 17:22], lm[1, 17:22], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 22:27], lm[1, 22:27], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 27:31], lm[1, 27:31], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 31:36], lm[1, 31:36], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 36:42], lm[1, 36:42], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 42:48], lm[1, 42:48], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 48:60], lm[1, 48:60], marker='o', markersize=2, linestyle='-', color='w', lw=1)
                ax.plot(lm[0, 60:68], lm[1, 60:68], marker='o', markersize=2, linestyle='-', color='w', lw=1)

        plt.savefig(file_to_save)