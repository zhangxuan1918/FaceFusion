import glob
import os

import tensorflow as tf
# Parse individual image from tfrecord file

def parse_tfrecord_tf(record, dtypes):
    features = tf.io.parse_single_example(record, features={
        'image': tf.io.FixedLenFeature([], tf.string),
        'shape_para': tf.io.FixedLenFeature([], tf.string),
        'pose_para': tf.io.FixedLenFeature([], tf.string),
        'exp_para': tf.io.FixedLenFeature([], tf.string),
        'color_para': tf.io.FixedLenFeature([], tf.string),
        'illum_para': tf.io.FixedLenFeature([], tf.string),
        'tex_para': tf.io.FixedLenFeature([], tf.string),
        'pt2d': tf.io.FixedLenFeature([], tf.string),
    })
    image = tf.io.parse_tensor(features['image'], out_type=dtypes['image'])
    shape_para = tf.io.parse_tensor(features['shape_para'], out_type=dtypes['shape_para'])
    pose_para = tf.io.parse_tensor(features['pose_para'], out_type=dtypes['pose_para'])
    exp_para = tf.io.parse_tensor(features['exp_para'], out_type=dtypes['exp_para'])
    color_para = tf.io.parse_tensor(features['color_para'], out_type=dtypes['color_para'])
    illum_para = tf.io.parse_tensor(features['illum_para'], out_type=dtypes['illum_para'])
    tex_para = tf.io.parse_tensor(features['tex_para'], out_type=dtypes['tex_para'])
    pt2d = tf.io.parse_tensor(features['pt2d'], out_type=dtypes['pt2d'])

    return image, {
        'shape_para': shape_para,
        'pose_para': pose_para,
        'exp_para': exp_para,
        'color_para': color_para,
        'illum_para': illum_para,
        'tex_para': tex_para,
        'pt2d': pt2d
    }


def parse_tfrecord_np(record, dtypes):
    image, labels = parse_tfrecord_tf(record, dtypes)
    labels_np = {key: value.numpy() for key, value in labels.items()}

    return image.numpy(), labels_np


# Dataset class that loads data from tfrecords files
class TFRecordDataset:
    def __init__(self,
                 tfrecord_dir,  # Directory containing a collection of tfrecords files.
                 resolution=None,  # Dataset resolution, None = autodetect.
                 label_file=None,  # Relative path of the labels file, None = autodetect.
                 max_label_size=0,  # 0 = no labels, 'full' = full labels, <int> = N first label components.
                 repeat=True,  # Repeat dataset indefinitely.
                 shuffle_mb=4096,  # Shuffle data within specified window (megabytes), 0 = disable shuffling.
                 prefetch_mb=2048,  # Amount of data to prefetch (megabytes), 0 = disable prefetching.
                 buffer_mb=256,  # Read buffer size (megabytes).
                 num_threads=2):  # Number of concurrent threads.
        self.tfrecord_dir = tfrecord_dir
        self.resolution = None
        self.resolution_log2 = None
        self.shape = []  # [height, width, channel]
        self.dtypes = {
            'image': tf.uint8,
            'pose_para': tf.float32,
            'exp_para': tf.float32,
            'color_para': tf.float32,
            'illum_para': tf.float32,
            'tex_para': tf.float32,
            'pt2d': tf.float32
        }

        self.dynamic_range = [0, 255]

        self._tf_datasets = None
        self._tf_iterator = None
        self._tf_minibatch_np = None
        self._cur_minibatch = -1

        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) == 1
        tfr_file = tfr_files[0]
        tfr_opt = tf.io.TFRecordOptions('')
        tfr_shapes = {}
        for record in tf.compat.v1.io.tf_record_iterator(tfr_file, tfr_opt)
            ex, ex_labels = parse_tfrecord_np(record, self.dtypes)
            tfr_shapes['image'] = ex.shape
            for key, value in ex_labels:
                tfr_shapes[key] = value.shape

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
            dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
            bytes_per_item =