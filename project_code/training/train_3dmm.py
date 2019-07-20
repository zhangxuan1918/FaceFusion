import os

import tensorflow as tf
from tensorflow.python import keras

from project_code.models.networks_linear_3dmm import FaceNetLinear3DMM
from project_code.morphable_model.model.morphable_model import FFTfMorphableModel
from project_code.training.data import setup_3dmm_data
from project_code.training.loss import loss_3dmm
from project_code.training.opt import compute_landmarks, render_batch, save_rendered_images_for_eval


def train_3dmm(
        numof_epochs: int,
        ckpt,
        manager,
        face_model: FaceNetLinear3DMM,
        bfm: FFTfMorphableModel,
        config,
        log_dir: str,
        eval_dir: str
):
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
    metric_train = keras.metrics.Mean(name='loss_train', dtype=tf.float32)
    metric_test = keras.metrics.Mean(name='loss_test', dtype=tf.float32)

    train_ds, test_ds = setup_3dmm_data()

    optimizer = tf.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1
    )

    for epoch in range(numof_epochs):
        for batch_id, value in enumerate(train_ds):
            if batch_id % 100 == 0:
                print('training: batch={0}'.format(batch_id))

            ckpt.step.assign_add(1)
            images, ground_truth = value

            with train_summary_writer.as_default():

                train_3dmm_one_step(
                    face_model=face_model,
                    bfm=bfm,
                    optimizer=optimizer,
                    images=images,
                    metric=metric_train,
                    loss_type=config.loss_type
                )

                if tf.equal(optimizer.iterations % config.log_freq, 0):
                    tf.summary.scalar('loss_train', metric_train.result(), step=optimizer.iterations)
                    metric_train.reset_states()

            if batch_id > 0 and batch_id % 100 == 0:
                print('evaluate on test dataset')
                with test_summary_writer.as_default():
                    test_3dmm_one_step(
                        face_model=face_model,
                        bfm=bfm,
                        images=images,
                        metric=metric_test,
                        batch_id=batch_id,
                        eval_dir=eval_dir,
                        loss_type=config.loss_type
                    )

                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


def train_3dmm_one_step(
        face_model: FaceNetLinear3DMM,
        bfm: FFTfMorphableModel,
        optimizer,
        images,
        metric,
        loss_type
):

    with tf.GradientTape() as gradient_type:
        est = face_model(images, training=True)

        images_rendered = render_batch(
            batch_angles_grad=est['pos'][:, 0:3],
            batch_saling=est['pose'][:, 6],
            batch_t3d=est['pose'][:, 3:6],
            batch_shape=est['shape'],
            batch_exp=est['exp'],
            batch_tex=est['tex'],
            batch_color=est['color'],
            batch_illum=est['illum'],
            image_size=224,
            bfm=bfm
        )
        G_loss = loss_3dmm(
            images=images,
            images_rendered=images_rendered,
            metric=metric,
            loss_type=loss_type
        )

        trainable_vars = face_model.model.trainable_vars
        train_gradient = gradient_type.gradient(G_loss, trainable_vars)
        optimizer.apply_gradients(zip(train_gradient, trainable_vars))


def test_3dmm_one_step(
        face_model: FaceNetLinear3DMM,
        bfm: FFTfMorphableModel,
        images,
        metric,
        loss_type,
        eval_dir: str,
        batch_id: int
):

    est = face_model(images, training=False)
    est['landmark'] = compute_landmarks(
        poses_param=est.get('pose'),
        shapes_param=est.get('shape'),
        exps_param=est.get('exp'),
        bfm=bfm
    )

    images_rendered = render_batch(
        batch_angles_grad=est['pos'][:, 0:3],
        batch_saling=est['pose'][:, 6],
        batch_t3d=est['pose'][:, 3:6],
        batch_shape=est['shape'],
        batch_exp=est['exp'],
        batch_tex=est['tex'],
        batch_color=est['color'],
        batch_illum=est['illum'],
        image_size=224,
        bfm=bfm
    )

    loss_3dmm(
        images=images,
        images_rendered=images_rendered,
        metric=metric,
        loss_type=loss_type
    )

    save_rendered_images_for_eval(
        images=images,
        rendered_images=images_rendered,
        landmarks=est['landmark'],
        eval_dir=eval_dir,
        batch_id=batch_id
    )
