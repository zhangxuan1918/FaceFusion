import os

import tensorflow as tf

from project_code.data_tools.data_util import recover_3dmm_params
from project_code.morphable_model.mesh.visualize import render_and_save
from project_code.training.landmarks_util import compute_landmarks


def split_3dmm_labels(labels):
    """
    split labels into different 3dmm params
    :param labels:
    :return:
    """
    # get different labels
    # Shape_Para: (199,)
    # Pose_Para: (7,)
    # Exp_Para: (29,)
    # Color_Para: (7,)
    # Illum_Para: (10,)
    # pt2d: (136, )
    # Tex_Para: (199,)

    shape_labels = labels[:, :199]
    pose_labels = labels[:, 199: 206]
    exp_labels = labels[:, 206: 235]
    color_labels = labels[:, 235: 242]
    illum_labels = labels[:, 242: 252]
    landmark_labels = labels[:, 252: 388]
    tex_labels = labels[:, 388:]

    return shape_labels, pose_labels, exp_labels, color_labels, illum_labels, landmark_labels, tex_labels


def loss_norm(est, label, loss_type):
    """
    compute l1 or l2 loss
    :param est: estimations
    :param label: true labels
    :param loss_type: l1 or l2
    :return:
    """
    if loss_type == 'l2':
        return tf.reduce_mean(tf.square(est - label))
    elif loss_type == 'l1':
        return tf.reduce_mean(tf.abs(est - label))
    else:
        raise Exception('unsupported loss_type={0}'.format(loss_type))


def train_one_step_helper(trainable_vars, train_val, train_est, optimizer, loss_type):
    """
    compute loss and update weights

    :param trainable_vars:
    :param train_val:
    :param train_est:
    :param optimizer:
    :param loss_type:
    :return:
    """
    with tf.GradientTape() as tape:
        train_loss = loss_norm(train_val, train_est, loss_type)
        train_gradient = tape.gradient(train_loss, trainable_vars)
        optimizer.apply_gradients(zip(train_gradient, trainable_vars))

        return train_loss


def supervised_3dmm_train_one_step(
        face_model,
        images,
        optimizer,
        labels,
        metric_loss_shape,
        metric_loss_pose,
        metric_loss_exp,
        metric_loss_color,
        metric_loss_illum,
        metric_loss_tex,
        metric_loss_landmark,
        render_image_size=224
):
    """
    fine tune 3dmm model one training step
    :param face_model:
    :param images:
    :param optimizer:
    :param labels:
    :param metric_loss_shape:
    :param metric_loss_pose:
    :param metric_loss_exp:
    :param metric_loss_color:
    :param metric_loss_illum:
    :param metric_loss_tex:
    :return:
    """

    # get different labels
    shape_train_val, pose_train_val, exp_train_val, color_train_val, illum_train_val, landmark_train_val, tex_train_val = \
        split_3dmm_labels(labels=labels)

    # train one step
    illum_train_est, color_train_est, tex_train_est, shape_train_est, exp_train_est, pose_train_est = \
        face_model(images, training=True)

    shape_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_shape_trainable_vars(),
        train_val=shape_train_val,
        train_est=shape_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_shape_loss_type()
    )

    pose_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_pose_trainable_vars(),
        train_val=pose_train_val,
        train_est=pose_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_pose_loss_type()
    )

    exp_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_exp_trainable_vars(),
        train_val=exp_train_val,
        train_est=exp_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_exp_loss_type()
    )

    color_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_color_trainable_vars(),
        train_val=color_train_val,
        train_est=color_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_color_loss_type()
    )

    illum_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_illum_trainable_vars(),
        train_val=illum_train_val,
        train_est=illum_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_illum_loss_type()
    )

    tex_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_tex_trainable_vars(),
        train_val=tex_train_val,
        train_est=tex_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_tex_loss_type()
    )

    # compute landmarks using shape and pose parameter
    landmark_train_est = compute_landmarks(shape=shape_train_est, pose=pose_train_est, output_size=render_image_size)
    landmark_train_loss = train_one_step_helper(
        trainable_vars=face_model.get_shape_trainable_vars() + face_model.get_pose_trainable_vars(),
        train_val=landmark_train_val,
        train_est=landmark_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_tex_loss_type()
    )

    metric_loss_shape.update_state(shape_train_loss)
    metric_loss_exp.update_state(exp_train_loss)
    metric_loss_pose.update_state(pose_train_loss)
    metric_loss_color.update_state(color_train_loss)
    metric_loss_illum.update_state(illum_train_loss)
    metric_loss_landmark.update_state(landmark_train_loss)
    metric_loss_tex.update_state(tex_train_loss)


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


def supervised_3dmm_test(
        test_image_label_ds,
        face_model,
        bfm,
        epoch,
        batch_id,
        step,
        render_image_size,
        original_image_size,
        save_eval_to_folder):
    print('evaluate on test dataset')

    shape_test_loss = 0.0
    pose_test_loss = 0.0
    exp_test_loss = 0.0
    color_test_loss = 0.0
    illum_test_loss = 0.0
    landmark_test_loss = 0.0
    tex_test_loss = 0.0

    for k, value in enumerate(test_image_label_ds):
        if k % 100 == 0:
            print('test batch {0}'.format(k))
        images, labels = value
        shape_test_val, pose_test_val, exp_test_val, color_test_val, illum_test_val, landmark_test_val, tex_test_val = \
            split_3dmm_labels(labels=labels)

        landmark_test_est, illum_test_est, color_test_est, tex_test_est, shape_test_est, exp_test_est, pose_test_est = \
            face_model(images, training=False)

        shape_test_loss += loss_norm(label=shape_test_val, est=shape_test_est,
                                     loss_type=face_model.get_shape_loss_type())
        pose_test_loss += loss_norm(label=pose_test_val, est=pose_test_est,
                                    loss_type=face_model.get_pose_loss_type())
        exp_test_loss += loss_norm(label=exp_test_val, est=exp_test_est,
                                   loss_type=face_model.get_exp_loss_type())
        color_test_loss += loss_norm(label=color_test_val, est=color_test_est,
                                     loss_type=face_model.get_color_loss_type())
        illum_test_loss += loss_norm(label=illum_test_val, est=illum_test_est,
                                     loss_type=face_model.get_illum_loss_type())
        tex_test_loss += loss_norm(label=tex_test_val, est=tex_test_est,
                                   loss_type=face_model.get_tex_loss_type())

        landmark_test_loss += loss_norm(label=landmark_test_val, est=landmark_test_est,
                                        loss_type=face_model.get_landmark_loss_type())

        if batch_id % 5000 == 0 and k == 0:
            # save sample rendered images
            for j in range(4):
                ## render with groudtruth data
                save_rendered_image_for_eval(
                    image=images[j],
                    bfm=bfm,
                    render_image_size=render_image_size,
                    original_image_size=original_image_size,
                    shape_param=shape_test_val,
                    pose_param=pose_test_val,
                    exp_param=exp_test_val,
                    color_param=color_test_val,
                    illum_param=illum_test_val,
                    tex_param=tex_test_val,
                    landmark_param=landmark_test_val,
                    save_eval_to_folder=save_eval_to_folder,
                    epoch=epoch,
                    batch_id=batch_id,
                    image_id=j
                )

                ## render with estimated data
                # save_rendered_image_for_eval(
                #     image=images[j],
                #     bfm=bfm,
                #     render_image_size=render_image_size,
                #     original_image_size=original_image_size,
                #     shape_test_est=shape_test_est,
                #     pose_test_est=pose_test_est,
                #     exp_test_est=exp_test_est,
                #     color_test_est=color_test_est,
                #     illum_test_est=illum_test_est,
                #     tex_test_est=tex_test_est,
                #     landmark_test_est=landmark_test_est,
                #     save_eval_to_folder=save_eval_to_folder,
                #     epoch = epoch,
                #     batch_id = batch_id,
                #     image_id = j
                # )
    print('======= epoch = {0}, batch={1}'.format(epoch, batch_id))
    print('shape loss: %5f' % shape_test_loss)
    print('pose loss: %5f' % pose_test_loss)
    print('exp loss: %5f' % exp_test_loss)
    print('color loss: %5f' % color_test_loss)
    print('illum loss: %5f' % illum_test_loss)
    print('landmark loss: %5f' % landmark_test_loss)
    print('tex loss: %5f' % tex_test_loss)

    tf.summary.scalar('loss_test_shape', shape_test_loss, step=step)
    tf.summary.scalar('loss_test_exp', exp_test_loss, step=step)
    tf.summary.scalar('loss_test_color', color_test_loss, step=step)
    tf.summary.scalar('loss_test_illum', illum_test_loss, step=step)
    tf.summary.scalar('loss_test_landmark', landmark_test_loss, step=step)
    tf.summary.scalar('loss_test_tex', tex_test_loss, step=step)
