import tensorflow as tf

from morphable_model.model.morphable_model import FFTfMorphableModel
from training.config_util import EasyDict
from training.data import setup_3dmm_warmup_data
from training.log import setup_summary
from training.loss import loss_3dmm_warmup
from training.opt import compute_landmarks, save_rendered_images_for_warmup_eval


def train_3dmm_warmup(
        ckpt,
        manager,
        face_model,
        bfm: FFTfMorphableModel,
        config: EasyDict,
        log_dir: str,
        eval_dir: str
):
    train_summary_writer, metric_train, test_summary_writer, metric_test = setup_summary(
        log_dir=log_dir
    )
    train_ds, test_ds = setup_3dmm_warmup_data(
        bfm=bfm,
        batch_size=config.batch_size,
        data_train_dir=config.data_train_dir,
        data_test_dir=config.data_test_dir
    )

    optimizer = tf.optimizers.Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1
    )

    loss_types = {
        'shape': config.loss_shape_type if hasattr(config, 'loss_shape_type') else 'l2',
        'pose': config.loss_pose_type if hasattr(config, 'loss_pose_type') else 'l2',
        'exp': config.loss_exp_type if hasattr(config, 'loss_exp_type') else 'l2',
        'color': config.loss_color_type if hasattr(config, 'loss_color_type') else 'l2',
        'illum': config.loss_illum_type if hasattr(config, 'loss_illum_type') else 'l2',
        'tex': config.loss_tex_type if hasattr(config, 'loss_tex_type') else 'l2',
        'landmark': config.loss_landmark_type if hasattr(config, 'loss_landmark_type') else 'l2',
    }

    loss_weights = {
        'shape': 20,
        'pose': 20,
        'exp': 20,
        'color': 5,
        'illum': 5,
        'tex': 5,
        'landmark': 10
    }
    is_use_loss_landmark = False
    for epoch in range(config.num_of_epochs):
        if epoch == 1:
            face_model.resnet50.unfreeze()
            face_model.summary()
            is_use_loss_landmark = True

        for batch_id, value in enumerate(train_ds):
            if batch_id % 100 == 0:
                print('warm up training: batch={0}'.format(batch_id))

            ckpt.step.assign_add(1)
            # ground_truth = \
            # {
            #     'shape': shape_gt,
            #     'pose': pose_gt,
            #     'exp': exp_gt,
            #     'color': color_gt,
            #     'illum': illum_gt,
            #     'tex': tex_gt,
            #     'landmark': lm_gt,
            # }
            images, ground_truth = value
            with train_summary_writer.as_default():

                train_3dmm_warmup_one_step(
                    face_model=face_model,
                    bfm=bfm,
                    optimizer=optimizer,
                    images=images,
                    ground_truth=ground_truth,
                    metric=metric_train,
                    loss_types=loss_types,
                    loss_weights=loss_weights,
                    epoch=epoch,
                    step_id=batch_id,
                    is_use_loss_landmark=is_use_loss_landmark
                )

                if tf.equal(optimizer.iterations % config.log_freq, 0):
                    for param, metric in metric_train.items():
                        tf.summary.scalar(param, metric.result(), step=optimizer.iterations)
                        metric.reset_states()

            # if batch_id % config.eval_freq == 0:
            if batch_id > 0 and batch_id % config.eval_freq == 0:
                print('evaluate on test dataset')
                with test_summary_writer.as_default():
                    test_3dmm_warmup_one_step(
                        face_model=face_model,
                        bfm=bfm,
                        test_ds=test_ds,
                        metric=metric_test,
                        loss_types=loss_types,
                        loss_weights=loss_weights,
                        render_image_size=224,
                        step_id=int(ckpt.step),
                        eval_dir=eval_dir,
                        is_use_loss_landmark=True
                    )

                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


def train_3dmm_warmup_one_step(
        face_model,
        bfm: FFTfMorphableModel,
        optimizer,
        images,
        ground_truth,
        metric: dict,
        loss_types: dict,
        loss_weights: dict,
        epoch: int,
        step_id: int,
        is_use_loss_landmark: bool
):
    with tf.GradientTape() as gradient_type:
        est = face_model(images, training=True)

        if is_use_loss_landmark:
            est['landmark'] = compute_landmarks(
                poses_param=est.get('pose') * bfm.stats_pose_std + bfm.stats_pose_mu,
                shapes_param=est.get('shape') * bfm.stats_shape_std + bfm.stats_shape_mu,
                exps_param=est.get('exp') * bfm.stats_exp_std + bfm.stats_exp_mu,
                bfm=bfm
            )

        G_loss, loss_info = loss_3dmm_warmup(
            gt=ground_truth,
            est=est,
            metric=metric,
            loss_types=loss_types,
            loss_weights=loss_weights,
            is_use_loss_landmark=is_use_loss_landmark
        )

        print('epoch: {epoch}/{step_id}, {loss}'.format(epoch=epoch, step_id=step_id, loss=loss_info))

        trainable_vars = face_model.model.trainable_variables
        train_gradient = gradient_type.gradient(G_loss, trainable_vars)
        optimizer.apply_gradients(zip(train_gradient, trainable_vars))


def test_3dmm_warmup_one_step(
        face_model,
        bfm: FFTfMorphableModel,
        test_ds,
        metric: dict,
        loss_types: dict,
        loss_weights: dict,
        render_image_size,
        eval_dir: str,
        step_id: int,
        is_use_loss_landmark: bool
):
    G_loss = 0
    for i, value in enumerate(test_ds):
        # ground_truth = \
        #     {
        #         'shape': shape_gt,
        #         'pose': pose_gt,
        #         'exp': exp_gt,
        #         'color': color_gt,
        #         'illum': illum_gt,
        #         'tex': tex_gt,
        #         'landmark': lm_gt,
        #     }
        images, ground_truth = value

        est = face_model(images, training=False)

        if is_use_loss_landmark:
            est['landmark'] = compute_landmarks(
                poses_param=est.get('pose') * bfm.stats_pose_std + bfm.stats_pose_mu,
                shapes_param=est.get('shape') * bfm.stats_shape_std + bfm.stats_shape_mu,
                exps_param=est.get('exp') * bfm.stats_exp_std + bfm.stats_exp_mu,
                bfm=bfm
            )

        one_loss, _ = loss_3dmm_warmup(
            gt=ground_truth,
            est=est,
            metric=metric,
            loss_types=loss_types,
            loss_weights=loss_weights,
            is_use_loss_landmark=is_use_loss_landmark
        )

        G_loss += one_loss

        if i == 0:
            save_rendered_images_for_warmup_eval(
                bfm=bfm,
                images=images,
                gt=ground_truth,
                est=est,
                image_size=render_image_size,
                eval_dir=eval_dir,
                batch_id=step_id
            )
    print('step={0}, test loss: {1}'.format(step_id, G_loss))
