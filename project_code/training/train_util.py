import tensorflow as tf

from project_code.training.landmarks_util import compute_landmarks


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


def train_one_step_landmarks_helper(
        trainable_vars,
        train_landmark_val,
        train_shape_est,
        train_pose_est,
        optimizer,
        loss_type,
        output_size=224
):
    """
    compute landmark loss and update weights

    :param trainable_vars:
    :param train_val:
    :param train_est:
    :param optimizer:
    :param loss_type:
    :param output_size:
    :return:
    """

    # compute 2d landmarks from shape param and pose param
    train_landmark_est = compute_landmarks(shape=train_shape_est, pose=train_pose_est, output_size=output_size)
    with tf.GradientTape() as tape:
        train_loss = loss_norm(train_landmark_est, train_landmark_val, loss_type)
        train_gradient = tape.gradient(train_loss, trainable_vars)
        optimizer.apply_gradients(zip(train_gradient, trainable_vars))

        return train_loss


def supervised_fine_tune_train_one_step(
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
        metric_loss_landmark
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
    # Shape_Para: (199,)
    # Pose_Para: (7,)
    # Exp_Para: (29,)
    # Color_Para: (7,)
    # Illum_Para: (10,)
    # pt2d: (136, )
    # Tex_Para: (199,)

    shape_train_val = labels[:, :199]
    pose_train_val = labels[:, 199: 206]
    exp_train_val = labels[:, 206: 235]
    color_train_val = labels[:, 235: 242]
    illum_train_val = labels[:, 242: 252]
    landmark_train_val = labels[:, 252: 388]
    tex_train_val = labels[:, 388:]

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

    landmark_train_loss = train_one_step_landmarks_helper(
        trainable_vars=face_model.get_shape_trainable_vars() + face_model.get_pose_trainable_vars(),
        train_landmark_val=landmark_train_val,
        train_shape_est=shape_train_est,
        train_pose_est=pose_train_est,
        optimizer=optimizer,
        loss_type=face_model.get_landmark_loss_type()
    )

    metric_loss_shape.update_state(shape_train_loss)
    metric_loss_exp.update_state(exp_train_loss)
    metric_loss_pose.update_state(pose_train_loss)
    metric_loss_color.update_state(color_train_loss)
    metric_loss_illum.update_state(illum_train_loss)
    metric_loss_landmark.update_state(landmark_train_loss)
    metric_loss_tex.update_state(tex_train_loss)
