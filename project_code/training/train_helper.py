import tensorflow as tf

from project_code.training.eval import save_rendered_image_for_eval
from project_code.training.loss import loss_norm
from project_code.training.opt import split_3dmm_labels, compute_landmarks


def train_one_step_helper(trainable_vars, train_val, train_est, optimizer, loss_type,
                          gradient_type):
    """
    compute loss and update weights

    :param trainable_vars:
    :param train_val:
    :param train_est:
    :param optimizer:
    :param loss_type:
    :return:
    """
    train_loss = loss_norm(train_val, train_est, loss_type)
    train_gradient = gradient_type.gradient(train_loss, trainable_vars)
    optimizer.apply_gradients(zip(train_gradient, trainable_vars))

    return train_loss


def supervised_3dmm_train_one_step(
        face_model,
        bfm,
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
        input_image_size
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
    :param metric_loss_landmark:
    :param input_image_size: used to compute landmarks 2d
    :return:
    """

    # get different labels
    shape_train_val, pose_train_val, exp_train_val, color_train_val, illum_train_val, landmark_train_val, tex_train_val = \
        split_3dmm_labels(labels=labels)

    with tf.GradientTape() as shape_tape, \
            tf.GradientTape() as pose_tape, \
            tf.GradientTape() as exp_tape, \
            tf.GradientTape() as color_tape, \
            tf.GradientTape() as illum_tape, \
            tf.GradientTape() as landmark_tape, \
            tf.GradientTape() as tex_tape:

        illum_train_est, color_train_est, tex_train_est, shape_train_est, exp_train_est, pose_train_est = \
            face_model(images, training=True)

        shape_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_shape_trainable_vars(),
            train_val=shape_train_val,
            train_est=shape_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_shape_loss_type(),
            gradient_type=shape_tape
        )

        pose_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_pose_trainable_vars(),
            train_val=pose_train_val,
            train_est=pose_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_pose_loss_type(),
            gradient_type=pose_tape
        )

        exp_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_exp_trainable_vars(),
            train_val=exp_train_val,
            train_est=exp_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_exp_loss_type(),
            gradient_type=exp_tape
        )

        color_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_color_trainable_vars(),
            train_val=color_train_val,
            train_est=color_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_color_loss_type(),
            gradient_type=color_tape
        )

        illum_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_illum_trainable_vars(),
            train_val=illum_train_val,
            train_est=illum_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_illum_loss_type(),
            gradient_type=illum_tape
        )

        tex_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_tex_trainable_vars(),
            train_val=tex_train_val,
            train_est=tex_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_tex_loss_type(),
            gradient_type=tex_tape
        )

        # compute landmarks using shape and pose parameter
        landmark_train_est = compute_landmarks(
            poses_param=pose_train_est,
            shapes_param=shape_train_est,
            exps_param=exp_train_est,
            bfm=bfm,
            output_size=input_image_size
        )
        landmark_train_loss = train_one_step_helper(
            trainable_vars=face_model.get_shape_trainable_vars() + face_model.get_pose_trainable_vars(),
            train_val=landmark_train_val,
            train_est=landmark_train_est,
            optimizer=optimizer,
            loss_type=face_model.get_tex_loss_type(),
            gradient_type=landmark_tape
        )

    metric_loss_shape.update_state(shape_train_loss)
    metric_loss_exp.update_state(exp_train_loss)
    metric_loss_pose.update_state(pose_train_loss)
    metric_loss_color.update_state(color_train_loss)
    metric_loss_illum.update_state(illum_train_loss)
    metric_loss_landmark.update_state(landmark_train_loss)
    metric_loss_tex.update_state(tex_train_loss)


def supervised_3dmm_test(
        test_image_label_ds,
        face_model,
        bfm,
        epoch,
        batch_id,
        step,
        render_image_size,
        input_image_size,
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

        illum_test_est, color_test_est, tex_test_est, shape_test_est, exp_test_est, pose_test_est = \
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

        landmark_test_est = compute_landmarks(
            poses_param=pose_test_est,
            shapes_param=shape_test_est,
            exps_param=exp_test_est,
            bfm=bfm,
            output_size=input_image_size
        )
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
                    input_image_size=input_image_size,
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
