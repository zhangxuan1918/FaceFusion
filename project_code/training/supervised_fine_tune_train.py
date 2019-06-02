import datetime
import os

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_fine_tune_labeled_data
from project_code.data_tools.data_util import recover_3dmm_params
from project_code.models.networks_3dmm import Face3DMM
from project_code.morphable_model.mesh.visualize import plot_rendered
from project_code.morphable_model.model.morphable_model import MorphableModel

tf.random.set_seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

tf.config.gpu.set_per_process_memory_fraction(0.9)
tf.config.gpu.set_per_process_memory_growth(True)

learning_rate = 0.001
batch_size = 4
image_size = 224
epochs = 10
save_to_folder = 'G:\PycharmProjects\FaceFusion\project_code\data\supervised_fine_tuned_model\{0}'.format(
    datetime.datetime.today().strftime('%Y-%m-%d')
)
save_model_to_folder = os.path.join(save_to_folder, 'models')
save_eval_to_folder = os.path.join(save_to_folder, 'eval')
render_image_height = render_image_width = 450

# create BFM
bfm = MorphableModel('G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\BFM\BFM.mat')

# load training dataset
training_data_root_folder = 'H:/300W-LP/300W_LP/'
training_image_label_ds = get_3dmm_fine_tune_labeled_data(
    data_root_folder=training_data_root_folder,
    suffix='*/*.jpg'
)

training_image_label_ds = training_image_label_ds.repeat()
training_image_label_ds = training_image_label_ds.batch(batch_size)
training_image_label_ds = training_image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(training_image_label_ds)

# load testing dataset
testing_data_root_folder = 'H:/300W-LP/300W_LP/HELEN/'
testing_image_label_ds = get_3dmm_fine_tune_labeled_data(
    data_root_folder=testing_data_root_folder,
    suffix='*.jpg'
)

testing_image_label_ds = testing_image_label_ds.repeat()
testing_image_label_ds = testing_image_label_ds.batch(batch_size)
testing_image_label_ds = testing_image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(testing_image_label_ds)

# load Face model
checkpoint_dir = 'G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190530/'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

face_model = Face3DMM()
face_model.load_weights(latest_checkpoint)

shape_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
pose_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
exp_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
color_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
illum_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
landmark_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)
tex_optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.5)

for epoch in range(1):
    for i, value in enumerate(training_image_label_ds):
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

        with tf.GradientTape() as shape_tape, \
                tf.GradientTape() as pose_tape, \
                tf.GradientTape() as exp_tape, \
                tf.GradientTape() as color_tape, \
                tf.GradientTape() as illum_tape, \
                tf.GradientTape() as landmark_tape, \
                tf.GradientTape() as tex_tape:

            landmark_test_est, illum_test_est, color_test_est, tex_test_est, shape_test_est, exp_test_est, pose_test_est = \
                face_model(images, training=True)

            shape_test_loss = tf.reduce_mean(tf.square(shape_test_val - shape_test_est))
            pose_test_loss = tf.reduce_mean(tf.square(pose_test_val - pose_test_est))
            exp_test_loss = tf.reduce_mean(tf.square(exp_test_val - exp_test_est))
            color_test_loss = tf.reduce_mean(tf.square(color_test_val - color_test_est))
            illum_test_loss = tf.reduce_mean(tf.square(illum_test_val - illum_test_est))
            landmark_test_loss = tf.reduce_mean(tf.square(landmark_test_val - landmark_test_est))
            tex_test_loss = tf.reduce_mean(tf.square(tex_test_val - tex_test_est))

            shape_gradient = shape_tape.gradient(shape_test_loss, face_model.get_shape_trainable_vars())
            pose_gradient = pose_tape.gradient(pose_test_loss, face_model.get_pose_trainable_vars())
            exp_gradient = exp_tape.gradient(exp_test_loss, face_model.get_exp_trainable_vars())
            color_gradient = color_tape.gradient(color_test_loss, face_model.get_color_trainable_vars())
            illum_gradient = illum_tape.gradient(illum_test_loss, face_model.get_illum_trainable_vars())
            landmark_gradient = landmark_tape.gradient(landmark_test_loss, face_model.get_landmark_trainable_vars())
            tex_gradient = tex_tape.gradient(tex_test_loss, face_model.get_tex_trainable_vars())

            shape_optimizer.apply_gradients(zip(shape_gradient, face_model.get_shape_trainable_vars()))
            pose_optimizer.apply_gradients(zip(pose_gradient, face_model.get_pose_trainable_vars()))
            exp_optimizer.apply_gradients(zip(exp_gradient, face_model.get_exp_trainable_vars()))
            color_optimizer.apply_gradients(zip(color_gradient, face_model.get_color_trainable_vars()))
            illum_optimizer.apply_gradients(zip(illum_gradient, face_model.get_illum_trainable_vars()))
            landmark_optimizer.apply_gradients(zip(landmark_gradient, face_model.get_landmark_trainable_vars()))
            tex_optimizer.apply_gradients(zip(tex_gradient, face_model.get_tex_trainable_vars()))

        if i % 100 == 0:
            shape_test_loss = 0.0
            pose_lose = 0.0
            exp_test_loss = 0.0
            color_test_loss = 0.0
            illum_test_loss = 0.0
            landmark_test_loss = 0.0
            tex_test_loss = 0.0
            for k, value in enumerate(testing_image_label_ds):
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

                if k == 0:
                    # plot fitting
                    for j in range(batch_size):
                        # plot est
                        image_test, shape_param_test, pose_param_test, exp_param_test, color_param_test, illum_param_test, tex_param_test, landmarks_test = recover_3dmm_params(
                            image=images[j],
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
                            original_image_size=image_size,
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
