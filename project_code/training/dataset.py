import glob
import os

import numpy as np
import tensorflow as tf

# Parse individual image from tfrecord file


def parse_tfrecord_tf(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)})
    data = tf.io.decode_raw(features['image'], tf.uint8)
    return tf.reshape(data, features['shape'])


def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value  # temporary pylint workaround # pylint: disable=no-member
    data = ex.features.feature['image'].bytes_list.value[0]  # temporary pylint workaround # pylint: disable=no-member
    return np.fromstring(data, np.uint8).reshape(shape)


# Dataset class that loads data from tfrecords files
class TFRecordDataset:
    def __init__(self,
                 tfrecord_dir,  # Directory containing a collection of tfrecords files.
                 resolution=None,  # Dataset resolution, None = autodetect.
                 label_file=None,  # Relative path of the labels file, None = autodetect.
                 max_label_size=0,  # 0 = no labels, 'full' = full labels, <int> = N first label components.
                 repeat=True,  # Repeat dataset indefinitely.
                 batch_size=64,  # batch size
                 shuffle_mb=4096,  # Shuffle data within specified window (megabytes), 0 = disable shuffling.
                 prefetch_mb=2048,  # Amount of data to prefetch (megabytes), 0 = disable prefetching.
                 buffer_mb=256,  # Read buffer size (megabytes).
                 num_threads=1,   # Number of concurrent threads.
                 num_gpu=1, # Number of gpu
                 ):  # Data distribution for multi gpu
        self.tfrecord_dir = tfrecord_dir
        self.resolution = resolution
        self.label_file = label_file
        self.label_size = None
        self.label_dtype = None
        self.shape = []  # [height, width, channel]
        self.dtype = 'uint8'
        self.dynamic_range = [0, 255]
        self._np_labels = None
        self._tf_datasets = None
        self._tf_iterator = None
        self._tf_batch_in = batch_size
        if num_gpu > 0:
            self._tf_global_batch_in = batch_size * num_gpu
        else:
            self._tf_global_batch_in = self._tf_batch_in

        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) == 1
        tfr_file = tfr_files[0]
        tfr_opt = tf.io.TFRecordOptions('')
        tfr_shape = []
        for record in tf.compat.v1.io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shape.append(parse_tfrecord_np(record).shape)
            break

        # load label files
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels.npy')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess
        else:
            raise Exception('No label file found')
        self._np_labels = np.load(self.label_file)

        if max_label_size != 'full' and self._np_labels.shape[0] > max_label_size:
            self._np_labels = self._np_labels[:max_label_size, :]
        self.label_size = self._np_labels.shape[0]
        self.label_dtype = self._np_labels.dtype.name

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._np_labels)
            dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb << 20)
            dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
            dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
            bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
            if shuffle_mb > 0:
                dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
            if repeat:
                dset = dset.repeat()
            if prefetch_mb > 0:
                dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
            self._tf_datasets = dset.batch(self._tf_global_batch_in)

    # Get next mini batch as TensorFlow expressions
    def get_minibatch_tf(self):
        return next(self._tf_iterator)

    def __next__(self):
        return next(self._tf_iterator)

    def reset_iterator(self, strategy):
        self._tf_iterator = iter(strategy.experimental_distribute_dataset(self._tf_datasets))
