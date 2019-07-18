import tensorflow as tf
import tensorflow.python.keras as keras


def setup_summary(log_dir):
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
    metric_train = {
        'shape': keras.metrics.Mean(name='loss_shape_train', dtype=tf.float32),
        'pose': keras.metrics.Mean(name='loss_pose_train', dtype=tf.float32),
        'exp': keras.metrics.Mean(name='loss_exp_train', dtype=tf.float32),
        'color': keras.metrics.Mean(name='loss_color_train', dtype=tf.float32),
        'illum': keras.metrics.Mean(name='loss_illum_train', dtype=tf.float32),
        'tex': keras.metrics.Mean(name='loss_tex_train', dtype=tf.float32),
        'landmark': keras.metrics.Mean(name='loss_landmark_train', dtype=tf.float32),
    }
    metric_test = {
        'shape': keras.metrics.Mean(name='loss_shape_test', dtype=tf.float32),
        'pose': keras.metrics.Mean(name='loss_pose_test', dtype=tf.float32),
        'exp': keras.metrics.Mean(name='loss_exp_test', dtype=tf.float32),
        'color': keras.metrics.Mean(name='loss_color_test', dtype=tf.float32),
        'illum': keras.metrics.Mean(name='loss_illum_test', dtype=tf.float32),
        'tex': keras.metrics.Mean(name='loss_tex_test', dtype=tf.float32),
        'landmark': keras.metrics.Mean(name='loss_landmark_test', dtype=tf.float32),
    }

    return train_summary_writer, metric_train, test_summary_writer, metric_test