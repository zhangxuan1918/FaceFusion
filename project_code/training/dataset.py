import glob
import os
from abc import abstractmethod, ABC
import tensorflow_graphics as tfg
import tensorflow_addons as tfa
import numpy as np
import tensorflow as tf


# Parse individual image from tfrecord file


def parse_tfrecord_tf_supervised(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)})
    data = tf.io.decode_raw(features['image'], tf.uint8)
    return tf.reshape(data, features['shape'])


def parse_tfrecord_np_supervised(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['image'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)


def parse_tfrecord_tf_unsupervised(record):
    features = tf.io.parse_single_example(record, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    mask = tf.io.decode_raw(features['mask'], tf.uint8)
    return tf.reshape(image, features['shape']), tf.reshape(mask, features['shape'])


def parse_tfrecord_np_unsupervised(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    image = ex.features.feature['image'].bytes_list.value[0]
    mask = ex.features.feature['mask'].bytes_list.value[0]
    return np.fromstring(image, np.uint8).reshape(shape), np.fromstring(mask, np.uint8).reshape(shape)


def _augment_img_lmk_random_rescale(img, lmks):
    pos = np.random.randint(low=0, high=112, size=2)
    target_size = 224 + pos[0] + pos[1]
    img = tf.image.crop_to_bounding_box(
        img, 112-pos[0], 112-pos[0], target_size, target_size
    )
    # rescale image
    img = tf.cast(tf.image.resize(img, (224, 224)), dtype=tf.uint8)
    lmks = (lmks * 446 - 112 + pos[0]) / target_size
    return img, lmks


def _augment_img_lmk_random_rotate(img, lmks):
    # angle = tf.random.uniform([], -30, 30) * np.pi / 180.
    #
    # # landmark rotation matrix
    # matrix = tf.convert_to_tensor(
    #     [[tf.cos(-angle), -tf.sin(-angle)],
    #      [tf.sin(-angle), tf.cos(-angle)]],
    #     dtype=tf.float32
    # )

    angle = tf.random.uniform([], minval=-30, maxval=30) * np.pi / 180
    matrix = tfg.geometry.transformation.rotation_matrix_2d.from_euler([-angle])

    # landmarks shape (136) -> (2, 68) -> (68, 2)
    lmks = tf.transpose(tf.reshape(lmks, (2, 68)), (1, 0)) * 224.
    center = tf.reshape(tf.convert_to_tensor([112, 112], dtype=tf.float32), (1, 2))
    lmks = lmks - center
    lmks = tfg.geometry.transformation.rotation_matrix_2d.rotate(
        lmks, matrix
    )
    lmks += center

    # landmarks shape (68, 2) -> (2, 68) -> (136)
    lmks = tf.reshape(tf.transpose(lmks, (1, 0)), [-1]) / 224.

    img = tfa.image.rotate(img, angles=angle, interpolation='BILINEAR')

    return img, lmks


def augment_ffhq_arg_dataset(img, lmks):
    # p = np.random.rand()
    p = tf.random.uniform([])
    if p < 0.5:
        # image
        #     pad pad pad pad pad
        #     pad             pad
        #     pad             pad
        #     pad   IMAGE     pad
        #     pad             pad
        #     pad pad pad pad pad
        # pad = 112
        # IMAGE = 224
        # image size: (112 + 224 + 112) x (112 + 224 + 112)
        # no crop and rescaling
        img = img[112:336, 112:336, :]
        lmks = lmks * 2.0 - 0.5
    else:
        # with augmentation: crop and rescaling
        # random crop
        # crop image
        #   - top left corner: (pos[0], pos[0])
        #   - bottom right corner: (pos[0] + 224 + pos[1], pos[0] + 224 + pos[1])
        img, lmks = _augment_img_lmk_random_rescale(img, lmks)

    # p2 = np.random.rand()
    p2 = tf.random.uniform([])
    if p2 < 0.5:
        # with augmentation: rotation
        img, lmks = _augment_img_lmk_random_rotate(img, lmks)

    return img, lmks


# Dataset class that loads data from tfrecords files
class TFRecordDataset(ABC):
    def __init__(self,
                 _sentinel=None,
                 tfrecord_dir=None,  # Directory containing a collection of tfrecords files.
                 resolution=None,  # Dataset resolution, None = autodetect.
                 repeat=True,  # Repeat dataset indefinitely.
                 batch_size=64,  # batch size
                 shuffle_mb=4096,  # Shuffle data within specified window (megabytes), 0 = disable shuffling.
                 prefetch_mb=2048,  # Amount of data to prefetch (megabytes), 0 = disable prefetching.
                 buffer_mb=256,  # Read buffer size (megabytes).
                 num_threads=1,  # Number of concurrent threads.
                 num_gpu=1,  # Number of gpu
                 strategy=None,  # distributing strategy,
                 is_augment=False,  # whether to augment data or not
                 ):  # Data distribution for multi gpu

        if _sentinel is not None:
            raise ValueError('Initialize `TFRecordDataset` with only named arguments')

        self.tfrecord_dir = tfrecord_dir
        self.resolution = resolution
        self.shape = []  # [height, width, channel]
        self.dtype = 'uint8'
        self.dynamic_range = [0, 255]
        self._tf_datasets = None
        self._tf_iterator = None
        self._tf_batch_in = batch_size
        self.repeat = repeat
        self.shuffle_mb = shuffle_mb
        self.prefetch_mb = prefetch_mb
        self.buffer_mb = buffer_mb
        self.num_threads = num_threads
        self.is_augment = is_augment

        if num_gpu > 1:
            self._tf_global_batch_in = self._tf_batch_in * num_gpu
        else:
            self._tf_global_batch_in = self._tf_batch_in

        self.strategy = strategy
        if self.strategy is None:
            self.strategy = tf.distribute.OneDeviceStrategy('/gpu:0')

        self.prepare_dataset()

    @abstractmethod
    def prepare_dataset(self):
        pass

    def get_minibatch_tf(self):
        return next(self._tf_iterator)

    def __next__(self):
        return next(self._tf_iterator)


class TFRecordDatasetSupervised(TFRecordDataset):
    def __init__(self, _sentinel=None, label_file=None, **kwargs):
        if _sentinel is not None:
            raise ValueError('Initialize `TFRecordDatasetSupervised` with Only named arguments')

        self.label_file = label_file
        self.label_size = None
        self.label_dtype = None
        self._np_labels = None

        super(TFRecordDatasetSupervised, self).__init__(**kwargs)

    def prepare_dataset(self):
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) == 1
        tfr_file = tfr_files[0]
        tfr_opt = tf.io.TFRecordOptions('')
        tfr_shape = []
        for record in tf.compat.v1.io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shape.append(parse_tfrecord_np_supervised(record).shape)
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

        assert self.label_file is not None, ValueError('No label file found: %s' % self.label_file)

        self._np_labels = np.load(self.label_file)

        self.label_size = self._np_labels.shape[0]
        self.label_dtype = self._np_labels.dtype.name

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):

            dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=self.buffer_mb << 20)
            dset = dset.map(parse_tfrecord_tf_supervised, num_parallel_calls=self.num_threads)

            _tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._np_labels)
            dset = tf.data.Dataset.zip((dset, _tf_labels_dataset))

            if self.is_augment:
                dset = dset.map(augment_ffhq_arg_dataset, num_parallel_calls=self.num_threads)

            bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
            if self.shuffle_mb > 0:
                dset = dset.shuffle(((self.shuffle_mb << 20) - 1) // bytes_per_item + 1)
            if self.repeat:
                dset = dset.repeat()
            if self.prefetch_mb > 0:
                dset = dset.prefetch(((self.prefetch_mb << 20) - 1) // bytes_per_item + 1)
            self._tf_datasets = dset.batch(self._tf_global_batch_in)
            self._tf_iterator = iter(self.strategy.experimental_distribute_dataset(self._tf_datasets))


class TFRecordDatasetUnsupervised(TFRecordDataset):

    def prepare_dataset(self):
        assert os.path.isdir(self.tfrecord_dir)
        tfr_files = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) == 1
        tfr_file = tfr_files[0]
        tfr_opt = tf.io.TFRecordOptions('')
        tfr_shape = []
        for record in tf.compat.v1.io.tf_record_iterator(tfr_file, tfr_opt):
            tmp_image, _ = parse_tfrecord_np_unsupervised(record)
            tfr_shape.append(tf.shape(tmp_image))
            break

        with tf.name_scope('Dataset'), tf.device('/cpu:0'):

            dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=self.buffer_mb << 20)
            dset = dset.map(parse_tfrecord_tf_unsupervised, num_parallel_calls=self.num_threads)

            bytes_per_item = np.prod(tfr_shape) * 2 * np.dtype(self.dtype).itemsize
            if self.shuffle_mb > 0:
                dset = dset.shuffle(((self.shuffle_mb << 20) - 1) // bytes_per_item + 1)
            if self.repeat:
                dset = dset.repeat()
            if self.prefetch_mb > 0:
                dset = dset.prefetch(((self.prefetch_mb << 20) - 1) // bytes_per_item + 1)
            self._tf_datasets = dset.batch(self._tf_global_batch_in)
            self._tf_iterator = iter(self.strategy.experimental_distribute_dataset(self._tf_datasets))
