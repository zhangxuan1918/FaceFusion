import datetime
import os
from distutils.dir_util import copy_tree

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.data.experimental import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_fine_tune_labeled_data_split
from project_code.models.networks_3dmm import Face3DMM
from project_code.morphable_model.model.morphable_model import MorphableModel
from project_code.training.train_util import supervised_3dmm_train_one_step, update_tf_summary, supervised_3dmm_test

tf.random.set_seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

tf.config.gpu.set_per_process_memory_fraction(0.9)
tf.config.gpu.set_per_process_memory_growth(True)


def create_train_fine_tuning_folder(save_to_folder, model_folder=None):
    save_model_to_folder = os.path.join(save_to_folder, 'models')
    if not os.path.exists(save_model_to_folder):
        os.makedirs(save_model_to_folder)
        if model_folder is not None:
            # train from pretrained model since there is no checkpoint
            # copy face_vgg2 as pretrained model
            copy_tree(model_folder, save_model_to_folder)
    save_eval_to_folder = os.path.join(save_to_folder, 'eval')
    if not os.path.exists(save_eval_to_folder):
        os.makedirs(save_eval_to_folder)
    save_summary_to_folder = os.path.join(save_to_folder, 'summary')
    save_train_summary_to_folder = os.path.join(save_summary_to_folder, 'train')
    if not os.path.exists(save_train_summary_to_folder):
        os.makedirs(save_train_summary_to_folder)
    save_test_summary_to_folder = os.path.join(save_summary_to_folder, 'test')
    if not os.path.exists(save_test_summary_to_folder):
        os.makedirs(save_test_summary_to_folder)

    return save_model_to_folder, save_eval_to_folder, save_train_summary_to_folder, save_test_summary_to_folder


def create_3dmm_data_pipeline(data_root_folder):
    # load training dataset
    train_image_label_ds, test_image_label_ds = get_3dmm_fine_tune_labeled_data_split(
        data_root_folder=data_root_folder,
        suffix='*/*.jpg',
        test_data_ratio=0.01
    )

    train_image_label_ds = train_image_label_ds.repeat()
    train_image_label_ds = train_image_label_ds.batch(batch_size)
    train_image_label_ds = train_image_label_ds.prefetch(buffer_size=AUTOTUNE)

    test_image_label_ds = test_image_label_ds.batch(batch_size)
    test_image_label_ds = test_image_label_ds.prefetch(buffer_size=AUTOTUNE)

    return train_image_label_ds, test_image_label_ds


def create_or_load_face_model(checkpoint_dir, max_checkpoints_to_keep, input_shape):
    # load Face model
    face_model = Face3DMM()
    face_model.build(input_shape=input_shape)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=face_model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=max_checkpoints_to_keep)
    ckpt.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    return face_model, manager, ckpt


def supervised_3dmm_train(
        data_root_folder,
        save_to_folder,
        model_folder,

        learning_rate,
        batch_size,
        numof_epochs,
        log_freq,

        max_checkpoints_to_keep=3,
        input_image_size=224,
        input_image_channel=3,
        render_image_size=450
):
    # create folder
    save_model_to_folder, save_eval_to_folder, save_train_summary_to_folder, save_test_summary_to_folder = \
        create_train_fine_tuning_folder(save_to_folder=save_to_folder, model_folder=model_folder)

    # prepare data pipeline
    train_image_label_ds, test_image_label_ds = create_3dmm_data_pipeline(data_root_folder=data_root_folder)

    # create or load face model
    face_model, manager, ckpt = create_or_load_face_model(
        checkpoint_dir=save_model_to_folder,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        input_shape=(None, input_image_size, input_image_size, input_image_channel)
    )
    face_model.freeze_resnet()
    print(face_model.summary())

    optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)

    train_summary_writer = tf.summary.create_file_writer(save_train_summary_to_folder)
    test_summary_writer = tf.summary.create_file_writer(save_test_summary_to_folder)

    metric_loss_shape = keras.metrics.Mean(name='loss_shape', dtype=tf.float32)
    metric_loss_pose = keras.metrics.Mean(name='loss_pose', dtype=tf.float32)
    metric_loss_exp = keras.metrics.Mean(name='loss_exp', dtype=tf.float32)
    metric_loss_color = keras.metrics.Mean(name='loss_color', dtype=tf.float32)
    metric_loss_illum = keras.metrics.Mean(name='loss_illum', dtype=tf.float32)
    metric_loss_tex = keras.metrics.Mean(name='loss_tex', dtype=tf.float32)
    metric_loss_landmark = keras.metrics.Mean(name='loss_landmark', dtype=tf.float32)

    for epoch in range(numof_epochs):
        for batch_id, value in enumerate(train_image_label_ds):
            if batch_id % 100 == 0:
                print('train batch {0}'.format(batch_id))
            ckpt.step.assign_add(1)

            images, labels = value

            with train_summary_writer.as_default():
                supervised_3dmm_train_one_step(
                    face_model=face_model,
                    images=images,
                    optimizer=optimizer,
                    labels=labels,
                    metric_loss_shape=metric_loss_shape,
                    metric_loss_pose=metric_loss_pose,
                    metric_loss_exp=metric_loss_shape,
                    metric_loss_color=metric_loss_color,
                    metric_loss_illum=metric_loss_illum,
                    metric_loss_tex=metric_loss_tex,
                    metric_loss_landmark=metric_loss_landmark
                )

                if tf.equal(optimizer.iterations % log_freq, 0):
                    update_tf_summary(var_name='loss_train_shape', metric=metric_loss_shape, step=optimizer.iterations)
                    update_tf_summary(var_name='loss_train_exp', metric=metric_loss_exp, step=optimizer.iterations)
                    update_tf_summary(var_name='loss_train_color', metric=metric_loss_color, step=optimizer.iterations)
                    update_tf_summary(var_name='loss_train_illum', metric=metric_loss_illum, step=optimizer.iterations)
                    update_tf_summary(var_name='loss_train_landmark', metric=metric_loss_landmark,
                                      step=optimizer.iterations)
                    update_tf_summary(var_name='loss_train_tex', metric=metric_loss_tex.result(),
                                      step=optimizer.iterations)

            if batch_id > 0 and batch_id % 2000 == 0:
                print('evaluate on test dataset')
                with test_summary_writer.as_default():
                    supervised_3dmm_test(
                        test_image_label_ds=test_image_label_ds,
                        face_model=face_model,
                        bfm=bfm,
                        epoch=epoch,
                        batch_id=batch_id,
                        step=optimizer.iterations,
                        render_image_size=render_image_size,
                        original_image_size=input_image_size,
                        save_eval_to_folder=save_eval_to_folder)

                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))


if __name__ == '__main__':
    # when render image, we render image as 450x450x3
    render_image_size = 450
    # input size for face model as 224 x 224 x3
    input_image_size = 224
    input_image_channel = 3
    learning_rate = 0.01
    batch_size = 4
    numof_epochs = 5
    max_checkpoints_to_keep = 3
    log_freq = 64

    yyyy_mm_dd = datetime.datetime.today().strftime('%Y-%m-%d')

    # path to load bfm model
    bfm_mat_path = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\BFM\BFM.mat'
    # path to load 3dmm data
    data_root_folder = 'H:/300W-LP/300W_LP/'
    # path to load trained model, can be vgg2 model or face model
    model_folder = 'G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190530/'
    # folder to save model, eval and summary
    save_to_folder = 'G:\PycharmProjects\FaceFusion\project_code\data\supervised_3dmm_model\{0}'.format(
        yyyy_mm_dd
    )

    bfm = MorphableModel(bfm_mat_path)

    supervised_3dmm_train(
        data_root_folder=data_root_folder,
        save_to_folder=save_to_folder,
        model_folder=model_folder,

        learning_rate=learning_rate,
        batch_size=batch_size,
        numof_epochs=numof_epochs,
        log_freq=log_freq,

        max_checkpoints_to_keep=max_checkpoints_to_keep,
        input_image_size=input_image_size,
        input_image_channel=input_image_channel,
        render_image_size=render_image_size
    )
