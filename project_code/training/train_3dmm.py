import os

import tensorflow as tf
from tensorflow.python import keras

from morphable_model.model.morphable_model import FFTfMorphableModel
from training.config_util import EasyDict
from training.data import setup_3dmm_data
from training.loss import loss_3dmm
from training.opt import compute_landmarks, render_batch, save_rendered_images_for_eval


def train_3dmm(
        ckpt,
        manager,
        face_model,
        bfm: FFTfMorphableModel,
        config: EasyDict,
        log_dir: str,
        eval_dir: str
):
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
    metric_train = keras.metrics.Mean(name='loss_train', dtype=tf.float32)
    metric_test = keras.metrics.Mean(name='loss_test', dtype=tf.float32)

    train_ds, test_ds = setup_3dmm_data(
        data_train_dir=config.data_train_dir,
        data_test_dir=config.data_test_dir,
        batch_size=config.batch_size
    )

    optimizer = tf.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1
    )

    for epoch in range(config.num_of_epochs):
        for batch_id, images in enumerate(train_ds):
            if batch_id % 100 == 0:
                print('training: batch={0}'.format(batch_id))

            ckpt.step.assign_add(1)

            with train_summary_writer.as_default():

                train_3dmm_one_step(
                    face_model=face_model,
                    bfm=bfm,
                    optimizer=optimizer,
                    images=images,
                    metric=metric_train,
                    loss_type=config.loss_type,
                    epoch=epoch,
                    step_id=batch_id
                )

                if tf.equal(optimizer.iterations % config.log_freq, 0):
                    tf.summary.scalar('loss_train', metric_train.result(), step=optimizer.iterations)
                    metric_train.reset_states()

            if batch_id > 0 and batch_id % config.eval_freq == 0:
            # if batch_id % 100 == 0:
                print('evaluate on test dataset')
                with test_summary_writer.as_default():
                    test_3dmm_one_step(
                        face_model=face_model,
                        bfm=bfm,
                        test_ds=test_ds,
                        metric=metric_test,
                        batch_id=batch_id,
                        eval_dir=eval_dir,
                        loss_type=config.loss_type
                    )

                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


def train_3dmm_one_step(
        face_model,
        bfm: FFTfMorphableModel,
        optimizer,
        images,
        metric,
        loss_type: str,
        epoch: int,
        step_id: int
):

    with tf.GradientTape() as gradient_type:
        est = face_model(images, training=True)

        est['pose'] = est['pose'] * bfm.stats_pose_std + bfm.stats_pose_mu
        est['shape'] = est['shape'] * bfm.stats_shape_std + bfm.stats_shape_mu
        est['exp'] = est['exp'] * bfm.stats_exp_std + bfm.stats_exp_mu
        est['tex'] = est['tex'] * bfm.stats_tex_std + bfm.stats_tex_mu
        est['color'] = est['color'] * bfm.stats_color_std + bfm.stats_color_mu
        est['illum'] = est['illum'] * bfm.stats_illum_std + bfm.stats_illum_mu

        images_rendered = render_batch(
            batch_angles_grad=est['pose'][:, 0, 0:3],
            batch_saling=est['pose'][:, 0, 6],
            batch_t3d=est['pose'][:, 0, 3:6],
            batch_shape=est['shape'],
            batch_exp=est['exp'],
            batch_tex=est['tex'],
            batch_color=est['color'],
            batch_illum=est['illum'],
            image_size=224,
            bfm=bfm
        )
        G_loss = loss_3dmm(
            face_vgg2=face_model.face_vgg2,
            images=images,
            images_rendered=images_rendered,
            metric=metric,
            loss_type=loss_type
        )

        print('epoch: {epoch}/{step_id}, loss={loss}'.format(epoch=epoch, step_id=step_id, loss=G_loss.numpy()))

        trainable_vars = face_model.model.trainable_variables
        train_gradient = gradient_type.gradient(G_loss, trainable_vars)
        optimizer.apply_gradients(zip(train_gradient, trainable_vars))


def test_3dmm_one_step(
        face_model,
        bfm: FFTfMorphableModel,
        test_ds,
        metric,
        loss_type,
        eval_dir: str,
        batch_id: int
):
    G_loss = 0

    for i, images in enumerate(test_ds):
        est = face_model(images, training=False)
        est['landmark'] = compute_landmarks(
            poses_param=est.get('pose') * bfm.stats_pose_std + bfm.stats_pose_mu,
            shapes_param=est.get('shape') * bfm.stats_shape_std + bfm.stats_shape_mu,
            exps_param=est.get('exp') * bfm.stats_exp_std + bfm.stats_exp_mu,
            bfm=bfm
        )

        images_rendered = render_batch(
            batch_angles_grad=est['pose'][:, 0, 0:3],
            batch_saling=est['pose'][:, 0, 6],
            batch_t3d=est['pose'][:, 0, 3:6],
            batch_shape=est['shape'],
            batch_exp=est['exp'],
            batch_tex=est['tex'],
            batch_color=est['color'],
            batch_illum=est['illum'],
            image_size=224,
            bfm=bfm
        )

        G_loss += loss_3dmm(
            face_vgg2=face_model.face_vgg2,
            images=images,
            images_rendered=images_rendered,
            metric=metric,
            loss_type=loss_type
        )

        if i == 0:
            save_rendered_images_for_eval(
                images=images,
                rendered_images=images_rendered,
                landmarks=est['landmark'],
                eval_dir=eval_dir,
                batch_id=batch_id
            )
