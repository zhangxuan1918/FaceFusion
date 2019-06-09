import datetime
import os
from distutils.dir_util import copy_tree

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.data.experimental import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_fine_tune_labeled_data_split
from project_code.data_tools.data_util import recover_3dmm_params
from project_code.models.networks_3dmm import Face3DMM
from project_code.morphable_model.mesh.visualize import plot_rendered
from project_code.morphable_model.model.morphable_model import MorphableModel
from project_code.training.train_util import supervised_fine_tune_train_one_step

tf.random.set_seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

tf.config.gpu.set_per_process_memory_fraction(0.9)
tf.config.gpu.set_per_process_memory_growth(True)

LOG_FREQ = 64
LEARNING_RATE = 0.01
BATCH_SIZE = 4
IMAGE_SIZE = 224
EPOCHS = 10
save_to_folder = 'G:\PycharmProjects\FaceFusion\project_code\data\supervised_fine_tuned_model\{0}'.format(
    datetime.datetime.today().strftime('%Y-%m-%d')
)
save_model_to_folder = os.path.join(save_to_folder, 'models')
if not os.path.exists(save_model_to_folder):
    os.makedirs(save_model_to_folder)
    # train from pretrained model since there is no checkpoint
    # copy face_vgg2 as pretrained model
    copy_tree('G:\PycharmProjects\FaceFusion\project_code\data\pretrained_model\\20190608', save_model_to_folder)
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
render_image_height = render_image_width = 450

# create BFM
bfm = MorphableModel('G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\BFM\BFM.mat')

# load training dataset
data_root_folder = 'H:/300W-LP/300W_LP/'
train_image_label_ds, test_image_label_ds = get_3dmm_fine_tune_labeled_data_split(
    data_root_folder=data_root_folder,
    suffix='*/*.jpg',
    test_data_ratio=0.01
)

train_image_label_ds = train_image_label_ds.repeat()
train_image_label_ds = train_image_label_ds.batch(BATCH_SIZE)
train_image_label_ds = train_image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(train_image_label_ds)

# test_image_label_ds = test_image_label_ds.repeat()
test_image_label_ds = test_image_label_ds.batch(BATCH_SIZE)
test_image_label_ds = test_image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(test_image_label_ds)

# load Face model
checkpoint_dir = 'G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190530/'
face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=face_model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

face_model.freeze_resnet()
print(face_model.summary())

optimizer = tf.optimizers.Adam(LEARNING_RATE, beta_1=0.5)

ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=face_model)
manager = tf.train.CheckpointManager(ckpt, save_model_to_folder, max_to_keep=3)

train_summary_writer = tf.summary.create_file_writer(save_train_summary_to_folder)
test_summary_writer = tf.summary.create_file_writer(save_test_summary_to_folder)

metric_loss_shape = keras.metrics.Mean(name='loss_shape', dtype=tf.float32)
metric_loss_pose = keras.metrics.Mean(name='loss_pose', dtype=tf.float32)
metric_loss_exp = keras.metrics.Mean(name='loss_exp', dtype=tf.float32)
metric_loss_color = keras.metrics.Mean(name='loss_color', dtype=tf.float32)
metric_loss_illum = keras.metrics.Mean(name='loss_illum', dtype=tf.float32)
metric_loss_tex = keras.metrics.Mean(name='loss_tex', dtype=tf.float32)
metric_loss_landmark = keras.metrics.Mean(name='loss_landmark', dtype=tf.float32)
for epoch in range(1):
    for i, value in enumerate(train_image_label_ds):
        if i % 100 == 0:
            print('train batch {0}'.format(i))
        ckpt.step.assign_add(1)

        images, labels = value
        supervised_fine_tune_train_one_step(
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

        if tf.equal(optimizer.iterations % LOG_FREQ, 0):
            tf.summary.scalar('loss_train_shape', metric_loss_shape.result(), step=optimizer.iterations)
            metric_loss_shape.reset_states()
            tf.summary.scalar('loss_train_exp', metric_loss_exp.result(), step=optimizer.iterations)
            metric_loss_exp.reset_states()
            tf.summary.scalar('loss_train_color', metric_loss_color.result(), step=optimizer.iterations)
            metric_loss_color.reset_states()
            tf.summary.scalar('loss_train_illum', metric_loss_illum.result(), step=optimizer.iterations)
            metric_loss_illum.reset_states()
            tf.summary.scalar('loss_train_landmark', metric_loss_landmark.result(), step=optimizer.iterations)
            metric_loss_landmark.reset_states()
            tf.summary.scalar('loss_train_tex', metric_loss_tex.result(), step=optimizer.iterations)
            metric_loss_tex.reset_states()

        if i > 0 and i % 2000 == 0:
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
                # Shape_Para: (199,)
                # Pose_Para: (7,)
                # Exp_Para: (29,)
                # Color_Para: (7,)
                # Illum_Para: (10,)
                # pt2d: (136, )
                # Tex_Para: (199,)
                shape_test_val = labels[:, :199]
                pose_test_val = labels[:, 199: 206]
                exp_test_val = labels[:, 206: 235]
                color_test_val = labels[:, 235: 242]
                illum_test_val = labels[:, 242: 252]
                landmark_test_val = labels[:, 252: 388]
                tex_test_val = labels[:, 388:]

                landmark_test_est, illum_test_est, color_test_est, tex_test_est, shape_test_est, exp_test_est, pose_test_est = \
                    face_model(images, training=False)

                shape_test_loss += tf.reduce_mean(tf.square(shape_test_val - shape_test_est))
                pose_test_loss += tf.reduce_mean(tf.square(pose_test_val - pose_test_est))
                exp_test_loss += tf.reduce_mean(tf.square(exp_test_val - exp_test_est))
                color_test_loss += tf.reduce_mean(tf.square(color_test_val - color_test_est))
                illum_test_loss += tf.reduce_mean(tf.square(illum_test_val - illum_test_est))
                landmark_test_loss += tf.reduce_mean(tf.square(landmark_test_val - landmark_test_est))
                tex_test_loss += tf.reduce_mean(tf.square(tex_test_val - tex_test_est))

                if i % 5000 == 0 and k == 0:
                    # plot fitting
                    for j in range(4):
                        # plot est
                        image_test, shape_param_test, pose_param_test, exp_param_test, color_param_test, illum_param_test, landmarks_test, tex_param_test = recover_3dmm_params(
                            image=tf.image.resize(images[j], [render_image_height, render_image_width]),
                            shape_param=shape_test_est[j],
                            pose_param=pose_test_est[j],
                            exp_param=exp_test_est[j],
                            color_param=color_test_est[j],
                            illum_param=illum_test_est[j],
                            tex_param=tex_test_est[j],
                            landmarks=landmark_test_est[j])

                        ## plot ground truth
                        # image_test, shape_param_test, pose_param_test, exp_param_test, color_param_test, illum_param_test, landmarks_test, tex_param_test = \
                        #     recover_3dmm_params(
                        #         image=tf.image.resize(images[j], [render_image_height, render_image_width]),
                        #         shape_param=shape_test_val[j],
                        #         pose_param=pose_test_val[j],
                        #         exp_param=exp_test_val[j],
                        #         color_param=color_test_val[j],
                        #         illum_param=illum_test_val[j],
                        #         tex_param=tex_test_val[j],
                        #         landmarks=landmark_test_val[j])

                        save_to_file = os.path.join(save_eval_to_folder, 'epoch_{0}_batch_{1}_image_{2}.jpg'.format(
                            epoch, i, j
                        ))

                        plot_rendered(
                            original_image=image_test,
                            bfm=bfm,
                            shape_param=shape_param_test,
                            exp_param=exp_param_test,
                            tex_param=tex_param_test,
                            color_param=color_param_test,
                            illum_param=illum_param_test,
                            pose_param=pose_param_test,
                            landmarks=landmarks_test,
                            h=render_image_height,
                            w=render_image_width,
                            original_image_size=IMAGE_SIZE,
                            file_to_save=save_to_file
                        )

            print('======= epoch = {0}, batch={1}'.format(epoch, i))
            print('shape loss: %5f' % shape_test_loss)
            print('pose loss: %5f' % pose_test_loss)
            print('exp loss: %5f' % exp_test_loss)
            print('color loss: %5f' % color_test_loss)
            print('illum loss: %5f' % illum_test_loss)
            print('landmark loss: %5f' % landmark_test_loss)
            print('tex loss: %5f' % tex_test_loss)

            tf.summary.scalar('loss_test_shape', shape_test_loss, step=shape_optimizer.iterations)
            tf.summary.scalar('loss_test_exp', exp_test_loss, step=shape_optimizer.iterations)
            tf.summary.scalar('loss_test_color', color_test_loss, step=shape_optimizer.iterations)
            tf.summary.scalar('loss_test_illum', illum_test_loss, step=shape_optimizer.iterations)
            tf.summary.scalar('loss_test_landmark', landmark_test_loss, step=shape_optimizer.iterations)
            tf.summary.scalar('loss_test_tex', tex_test_loss, step=shape_optimizer.iterations)

            # face_model.save_weights(os.path.join(save_model_to_folder, 'face_model_{epoch}_{batch}.ckpt'.format(
            #     epoch=epoch,
            #     batch=i
            # )))
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
