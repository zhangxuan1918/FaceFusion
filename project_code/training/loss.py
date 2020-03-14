import tensorflow as tf
from tf_3dmm.mesh.reader import render_batch


def get_reconstruct_loss_fn(image_size, loss_factor=1.0):
    def regression_loss(gt_images, gt_params, est_params, tf_bfm, batch_size):
        # gt_params: EasyDict
        # est_params: EasyDict

        # geo loss
        est_geo_gt_pose_images = render_batch(
            pose_param=gt_params.pose_para,
            shape_param=est_params.shape_para,
            exp_param=est_params.exp_para,
            tex_param=est_params.tex_para,
            color_param=est_params.color_para,
            illum_param=est_params.illum_para,
            frame_width=image_size,
            frame_height=image_size,
            tf_bfm=tf_bfm,
            batch_size=batch_size
        )
        loss_pose = tf.reduce_mean(tf.square(gt_images - est_geo_gt_pose_images))

        # pose loss
        gt_geo_est_pose_images = render_batch(
            pose_param=est_params.pose_para,
            shape_param=gt_params.shape_para,
            exp_param=gt_params.exp_para,
            tex_param=egt_params.tex_para,
            color_param=gt_params.color_para,
            illum_param=gt_params.illum_para,
            frame_width=image_size,
            frame_height=image_size,
            tf_bfm=tf_bfm,
            batch_size=batch_size
        )
        loss_geo = tf.reduce_mean(tf.square(gt_images - gt_geo_est_pose_images))

        coef = tf.constant(loss_geo / (loss_pose + loss_geo))
        return loss_factor  *(coef * loss_pose + (1 - coef) * loss_geo)

    return regression_loss

