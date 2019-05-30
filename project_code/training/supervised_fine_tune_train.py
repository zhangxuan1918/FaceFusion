import os

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_fine_tune_labeled_data
from project_code.models.networks_3dmm import Face3DMM

tf.random.set_seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

tf.config.gpu.set_per_process_memory_fraction(0.9)
tf.config.gpu.set_per_process_memory_growth(True)

learning_rate = 0.001
batch_size = 4
image_size = 224
epochs = 10

# load dataset
data_root_folder = 'H:/300W-LP/300W_LP/'
image_label_ds = get_3dmm_fine_tune_labeled_data(
    data_root_folder=data_root_folder
)

image_label_ds = image_label_ds.repeat()
image_label_ds = image_label_ds.batch(batch_size)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(image_label_ds)

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
    for i, value in enumerate(image_label_ds):
        images, labels = value
        # Shape_Para: (199,)
        # Pose_Para: (7,)
        # Exp_Para: (29,)
        # Color_Para: (7,)
        # Illum_Para: (10,)
        # pt2d: (136, )
        # Tex_Para: (199,)
        shape_val = labels[:, :199]
        pose_val = labels[:, 199: 206]
        exp_val = labels[:, 206: 235]
        color_val = labels[:, 235: 242]
        illum_val = labels[:, 242: 252]
        landmark_val = labels[:, 252: 388]
        tex_val = labels[:, 388:]

        with tf.GradientTape() as shape_tape, \
                tf.GradientTape() as pose_tape, \
                tf.GradientTape() as exp_tape, \
                tf.GradientTape() as color_tape, \
                tf.GradientTape() as illum_tape, \
                tf.GradientTape() as landmark_tape, \
                tf.GradientTape() as tex_tape:

            landmark_est, illum_est, color_est, tex_est, shape_est, exp_est, pose_est = face_model(images,
                                                                                                   training=True)

            shape_loss = tf.reduce_mean(tf.square(shape_val - shape_est))
            pose_loss = tf.reduce_mean(tf.square(pose_val - pose_est))
            exp_loss = tf.reduce_mean(tf.square(exp_val - exp_est))
            color_loss = tf.reduce_mean(tf.square(color_val - color_est))
            illum_loss = tf.reduce_mean(tf.square(illum_val - illum_est))
            landmark_loss = tf.reduce_mean(tf.square(landmark_val - landmark_est))
            tex_loss = tf.reduce_mean(tf.square(tex_val - tex_est))

            shape_gradient = shape_tape.gradient(shape_loss, face_model.get_shape_trainable_vars())
            pose_gradient = pose_tape.gradient(pose_loss, face_model.get_pose_trainable_vars())
            exp_gradient = exp_tape.gradient(exp_loss, face_model.get_exp_trainable_vars())
            color_gradient = color_tape.gradient(color_loss, face_model.get_color_trainable_vars())
            illum_gradient = illum_tape.gradient(illum_loss, face_model.get_illum_trainable_vars())
            landmark_gradient = landmark_tape.gradient(landmark_loss, face_model.get_landmark_trainable_vars())
            tex_gradient = tex_tape.gradient(tex_loss, face_model.get_tex_trainable_vars())

            shape_optimizer.apply_gradients(zip(shape_gradient, face_model.get_shape_trainable_vars()))
            pose_optimizer.apply_gradients(zip(pose_gradient, face_model.get_pose_trainable_vars()))
            exp_optimizer.apply_gradients(zip(exp_gradient, face_model.get_exp_trainable_vars()))
            color_optimizer.apply_gradients(zip(color_gradient, face_model.get_color_trainable_vars()))
            illum_optimizer.apply_gradients(zip(illum_gradient, face_model.get_illum_trainable_vars()))
            landmark_optimizer.apply_gradients(zip(landmark_gradient, face_model.get_landmark_trainable_vars()))
            tex_optimizer.apply_gradients(zip(tex_gradient, face_model.get_tex_trainable_vars()))

        if i % 100 == 0:
            print('======= epoch = {0}, batch={1}'.format(epoch, i))
            print('shape loss: %5f' % shape_loss)
            print('pose loss: %5f' % pose_loss)
            print('exp loss: %5f' % exp_loss)
            print('color loss: %5f' % color_loss)
            print('illum loss: %5f' % illum_loss)
            print('landmark loss: %5f' % landmark_loss)
            print('tex loss: %5f' % tex_loss)
