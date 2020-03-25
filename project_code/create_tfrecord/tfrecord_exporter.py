import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class TFRecordExporter(ABC):

    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.tfr_writer = None
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print('Creating dataset \'{0}\''.format(tfrecord_dir))
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flusing data...', end='', flush=True)
        if self.tfr_writer is not None:
            self.tfr_writer.close()
            self.tfr_writer = None
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Add %d images.' % self.cur_images)

    def choose_shuffled_order(self, random_shuffle, seed=1987):
        order = np.arange(self.expected_images)
        if random_shuffle:
            np.random.RandomState(seed).shuffle(order)
        return order

    @abstractmethod
    def add_image(self, **kwargs):
        pass

    @abstractmethod
    def add_labels(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TFRecordExporterSupervised(TFRecordExporter):

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images))

        if self.shape is None:
            self.shape = img.shape
            assert self.shape[2] == 3
            assert self.shape[0] == self.shape[1]

            tfr_opt = tf.io.TFRecordOptions('')
            tfr_file = self.tfr_prefix + '.tfrecords'
            self.tfr_writer = tf.io.TFRecordWriter(tfr_file, tfr_opt)

        assert img.shape == self.shape

        quant = np.rint(img).clip(0, 255).astype(np.uint8)
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
        }))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        np.save(self.tfr_prefix + '-rrx.labels', labels.astype(np.float32))


class TFRecordExporterUnsupervised(TFRecordExporter):

    def add_image(self, img, mask):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images))

        if self.shape is None:
            self.shape = img.shape
            assert self.shape[2] == 3
            assert self.shape[0] == self.shape[1]

            tfr_opt = tf.io.TFRecordOptions('')
            tfr_file = self.tfr_prefix + '.tfrecords'
            self.tfr_writer = tf.io.TFRecordWriter(tfr_file, tfr_opt)

        assert img.shape == self.shape
        assert mask.shape == self.shape

        quant_image = np.rint(img).clip(0, 255).astype(np.uint8)
        quant_mask = np.rint(mask).clip(0, 255).astype(np.uint8)

        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant_image.shape)),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant_image.tostring()])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant_mask.tostring()])),
        }))
        self.tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        raise NotImplementedError('No labels should be supplied in unsupervised learning')