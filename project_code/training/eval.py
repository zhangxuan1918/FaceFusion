import os

import tensorflow as tf

from project_code.data_tools.data_util import recover_3dmm_params
from project_code.morphable_model.mesh.visualize import render_and_save


def update_tf_summary(var_name, metric, step):
    tf.summary.scalar(var_name, metric.result(), step=step)
    metric.reset_states()


def save_rendered_image_for_eval(
        image,
        bfm,
        render_image_size,
        original_image_size,
        shape_param,
        pose_param,
        exp_param,
        color_param,
        illum_param,
        tex_param,
        landmark_param,
        save_eval_to_folder,
        epoch,
        batch_id,
        image_id
):
    # plot est
    image_test, shape_param_test, pose_param_test, exp_param_test, color_param_test, illum_param_test, \
        landmarks_test, tex_param_test = recover_3dmm_params(
            image=tf.image.resize(image, [render_image_size, render_image_size]),
            shape_param=shape_param,
            pose_param=pose_param,
            exp_param=exp_param,
            color_param=color_param,
            illum_param=illum_param,
            tex_param=tex_param,
            landmarks=landmark_param)

    save_to_file = os.path.join(save_eval_to_folder, 'epoch_{epoch}_batch_{batch_id}_image_{image_id}.jpg'.format(
        epoch=epoch, batch_id=batch_id, image_id=image_id
    ))

    render_and_save(
        original_image=image_test,
        bfm=bfm,
        shape_param=shape_param_test,
        exp_param=exp_param_test,
        tex_param=tex_param_test,
        color_param=color_param_test,
        illum_param=illum_param_test,
        pose_param=pose_param_test,
        landmarks=landmarks_test,
        h=render_image_size,
        w=render_image_size,
        original_image_size=original_image_size,
        file_to_save=save_to_file
    )