from pathlib import Path
import scipy.io as sio
import PIL
import glob
import os

import numpy as np
import scipy
import tensorflow as tf


class TFRecordExporter:

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

    def choose_shuffled_order(self):
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images))

        if self.shape is None:
            self.shape = img.shape
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]

            tfr_opt = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
            tfr_file = self.tfr_prefix + '.tfrecords'
            self.tfr_writer = tf.io.TFRecordWriter(tfr_file, tfr_opt)

        assert img.shape == self.shape
        quant = np.rint(img).clip(0, 255).astype(np.uint8)
        ex = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
            'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))
        }))
        self.tfr_writer.write(ex.SerializeToString())

        self.cur_images += 1

    def add_label(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_tfrecord(tfrecord_dir, image_filenames, img_shape, print_progress, progress_interval):
    labels = []
    mat_keys = ['roi', 'Shape_Para', 'Pose_Para', 'Exp_Para', 'Color_Para', 'Illum_Para', 'pt2d', 'Tex_Para']
    with TFRecordExporter(tfrecord_dir=tfrecord_dir,
                          expected_images=len(image_filenames),
                          print_progress=print_progress,
                          progress_interval=progress_interval) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            # load image
            img_filename = image_filenames[order[idx]]
            img = np.asarray(PIL.Image.open(img_filename))
            assert img.shape == img_shape
            img = img.transpose(2, 0, 1)
            tfr.add_image(img)

            # load mat
            # TODO: double check how to structure the label
            # TODO: now we only concatenate the params together as a numpy array
            mat_filename = '.'.join(img_filename.split('.')[:-1]) + '.mat'
            mat = sio.loadmat(mat_filename)
            tmp_list = []
            for key in mat_keys:
                tmp_list.append(mat[key].reshape(1, -1))
            labels.append(np.concatenate(*tmp_list))
    tfr.add_label(labels=np.asarray(labels))


def create_afw(tfrecord_dir, afw_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % afw_dir)
    glob_pattern = os.path.join(afw_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 5207
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, afw_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


def create_afw_flip(tfrecord_dir, afw_flip_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % afw_flip_dir)
    glob_pattern = os.path.join(afw_flip_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 5207
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, afw_flip_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


def create_helen(tfrecord_dir, helen_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % helen_dir)
    glob_pattern = os.path.join(helen_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 37676
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, helen_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


def create_helen_flip(tfrecord_dir, helen_flip_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % helen_flip_dir)
    glob_pattern = os.path.join(helen_flip_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 37676
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, helen_flip_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


def create_ibug(tfrecord_dir, ibug_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % ibug_dir)
    glob_pattern = os.path.join(ibug_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 1786
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, ibug_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


def create_ibug_flip(tfrecord_dir, ibug_flip_dir, print_progress, progress_interval):
    print('Loading AFW from \'%s\'' % ibug_flip_dir)
    glob_pattern = os.path.join(ibug_flip_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 1786
    img_shape = (450, 450, 3)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, ibug_flip_dir))
    create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, img_shape=img_shape,
                    print_progress=print_progress, progress_interval=progress_interval)


if __name__ == '__main__':
    print_progress = True,
    progress_interval = 100

    #G:/data/FaceFusion/3DMM/AFW/
    tfrecord_dir = Path('G:/data/FaceFusion/3DMM/AFW/')
    # H:/300W-LP/300W_LP/AFW/
    raw_data_dir = Path('H:/300W-LP/300W_LP/AFW/')
    create_afw(
        tfrecord_dir=tfrecord_dir,
        afw_dir=raw_data_dir,
        print_progress=print_progress,
        progress_interval=progress_interval
    )
