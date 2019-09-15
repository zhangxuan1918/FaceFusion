import os

import numpy as np
import tensorflow as tf
from tf_3dmm.mesh.render import render_2
from tf_3dmm.mesh.transform import affine_transform
import matplotlib.pyplot as plt

from data_tools.data_const import face_vgg2_input_mean
from morphable_model.model.morphable_model import FFTfMorphableModel


def random_translate(images, ground_truth, batch_size, target_size, bfm: FFTfMorphableModel):
    # random translate images
    tx = tf.random.uniform(
        shape=[batch_size],
        minval=0,
        maxval=32,
        dtype=tf.dtypes.int32
    )
    ty = tf.random.uniform(
        shape=[batch_size],
        minval=0,
        maxval=32,
        dtype=tf.dtypes.int32
    )

    delta_m = tf.zeros([batch_size, 7])
    delta_m[:, 6] = tf.math.divide(ty, bfm.stats_pose_mu[6])
    delta_m[:, 7] = tf.math.divide(tx, bfm.stats_pose_mu[7])

    ground_truth['pose'] = ground_truth['pose'] - delta_m
    images = tf.image.crop_to_bounding_box(images, tx, ty, target_size, target_size)
    return images, ground_truth


def compute_landmarks(
        poses_param,
        shapes_param,
        exps_param,
        bfm: FFTfMorphableModel
):
    start = 0
    end = shapes_param.shape[0]
    batch_landmarks = []
    kpt_indices = tf.expand_dims(bfm.get_landmark_indices(), axis=1)

    def cond(i, n):
        return i < n

    def body(i, n):
        vertices = bfm.get_vertices(shapes_param[i], exps_param[i])
        transformed_vertices = affine_transform(vertices, poses_param[i][0, 6], poses_param[i][0, 0:3],
                                                poses_param[i][0, 3:6])
        lm_3d = tf.gather_nd(transformed_vertices, kpt_indices)
        lm_2d = tf.concat(
            [tf.expand_dims(lm_3d[:, 0], axis=0), 224 - tf.expand_dims(lm_3d[:, 1], axis=0) - 1],
            axis=0)
        batch_landmarks.append(lm_2d)
        return i + 1, n

    tf.while_loop(cond=cond, body=body, loop_vars=[start, end])

    landmarks_2d = tf.stack(batch_landmarks, axis=0)

    tf.debugging.assert_shapes({landmarks_2d: (end, 2, tf.shape(kpt_indices)[0])})

    return landmarks_2d


def render_batch(
        batch_angles_grad,
        batch_saling,
        batch_t3d,
        batch_shape,
        batch_exp,
        batch_tex,
        batch_color,
        batch_illum,
        image_size,
        bfm
):
    start = 0
    end = batch_angles_grad.shape[0]
    rendered = []

    def cond(i, n):
        return i < n

    def body(i, n):
        image = render_2(
            angles_grad=batch_angles_grad[i],
            scaling=batch_saling[i],
            t3d=batch_t3d[i],
            shape_param=batch_shape[i],
            exp_param=batch_exp[i],
            tex_param=batch_tex[i],
            color_param=batch_color[i],
            illum_param=batch_illum[i],
            frame_width=image_size,
            frame_height=image_size,
            tf_bfm=bfm
        )
        rendered.append(image)
        return i + 1, n

    tf.while_loop(cond=cond, body=body, loop_vars=[start, end])

    images = tf.stack(rendered, axis=0)
    tf.debugging.assert_shapes({images: (end, image_size, image_size, 3)})
    return images


def save_rendered_images_for_warmup_eval(
        bfm: FFTfMorphableModel,
        images,
        gt,
        est,
        image_size,
        eval_dir,
        batch_id,
        num_images_to_render=4,
        max_images_in_dir=10,
):
    clean_up(data_folder=eval_dir, max_num_files=max_images_in_dir)
    # recover original input
    images += face_vgg2_input_mean

    # recover params
    est['pose'] = est['pose'] * bfm.stats_pose_std + bfm.stats_pose_mu
    est['shape'] = est['shape'] * bfm.stats_shape_std + bfm.stats_shape_mu
    est['exp'] = est['exp'] * bfm.stats_exp_std + bfm.stats_exp_mu
    est['tex'] = est['tex'] * bfm.stats_tex_std + bfm.stats_tex_mu
    est['color'] = est['color'] * bfm.stats_color_std + bfm.stats_color_mu
    est['illum'] = est['illum'] * bfm.stats_illum_std + bfm.stats_illum_mu

    images_est = render_batch(
        batch_angles_grad=est['pose'][0:num_images_to_render, 0, 0:3],
        batch_saling=est['pose'][0:num_images_to_render, 0, 6],
        batch_t3d=est['pose'][0:num_images_to_render, 0, 3:6],
        batch_shape=est['shape'][0:num_images_to_render],
        batch_exp=est['exp'][0:num_images_to_render],
        batch_tex=est['tex'][0:num_images_to_render],
        batch_color=est['color'][0:num_images_to_render],
        batch_illum=est['illum'][0:num_images_to_render],
        image_size=image_size,
        bfm=bfm
    )

    for i in range(num_images_to_render):
        image_gt = images[i].numpy().astype(np.uint8)

        image_est = images_est[i].numpy().astype(np.uint8)

        filename = os.path.join(eval_dir, 'batch_id_{batch_id}_rendered_{i}.jpg'.format(batch_id=batch_id, i=i))
        save_images(
            images=[image_gt, image_est],
            landmarks=[gt['landmark'][i].numpy(), est['landmark'][i].numpy()],
            titles=['ground_truth', 'estimation'],
            filename=filename
        )


def save_rendered_images_for_eval(
        images,
        rendered_images,
        landmarks,
        eval_dir,
        batch_id,
        num_images_to_render=4,
        max_images_in_dir=10,
):
    clean_up(data_folder=eval_dir, max_num_files=max_images_in_dir)
    images += face_vgg2_input_mean
    for i in range(num_images_to_render):
        image_gt = images[i].numpy().astype(np.uint8)

        image_est = rendered_images[i].numpy().astype(np.uint8)

        filename = os.path.join(eval_dir, '{batch_id}_rendered_{i}.jpg'.format(batch_id=batch_id, i=i))
        save_images(
            images=[image_gt, image_est],
            landmarks=[None, landmarks[i]],
            titles=['ground_truth', 'estimation'],
            filename=filename
        )


def clean_up(data_folder, max_num_files):
    # only keep 10 results

    files = os.listdir(data_folder)
    if len(files) <= max_num_files:
        return

    files = [os.path.join(data_folder, basename) for basename in files]
    files.sort(key=os.path.getctime)
    files = files[0: -max_num_files]
    for file in files:
        os.remove(file)


def save_images(images, filename, titles, landmarks=None):
    n = len(images)
    if landmarks is None:
        landmarks = [None] * n

    fig = plt.figure()

    for i in range(n):
        im = images[i]
        t = titles[i]
        lm = landmarks[i]

        ax = fig.add_subplot(1, n, i + 1)
        ax.set_ylim(bottom=224, top=0)
        ax.set_xlim(left=0, right=224)
        ax.imshow(im)
        if lm is not None:
            ax.plot(lm[0, 0:17], lm[1, 0:17], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 17:22], lm[1, 17:22], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 22:27], lm[1, 22:27], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 27:31], lm[1, 27:31], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 31:36], lm[1, 31:36], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 36:42], lm[1, 36:42], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 42:48], lm[1, 42:48], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 48:60], lm[1, 48:60], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(lm[0, 60:68], lm[1, 60:68], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)

        ax.set_title(t)
        plt.savefig(filename)
    plt.close(fig)
