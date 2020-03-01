import json
import sys
from pathlib import Path
import scipy.io as sio
import PIL
import glob
import os

import numpy as np
import scipy
import tensorflow as tf

from create_tfrecord.export_tfrecord_util import fn_extract_300W_LP_labels


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

    def choose_shuffled_order(self, random_shuffle):
        order = np.arange(self.expected_images)
        if random_shuffle:
            np.random.RandomState(123).shuffle(order)
        return order

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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_tfrecord(tfrecord_dir, image_filenames, image_size, process_label_fn, print_progress, progress_interval,
                    resolution=224, label_size=430, random_shuffle=True):
    with TFRecordExporter(tfrecord_dir=tfrecord_dir,
                          expected_images=len(image_filenames),
                          print_progress=print_progress,
                          progress_interval=progress_interval) as tfr:
        order = tfr.choose_shuffled_order(random_shuffle=random_shuffle)
        labels = np.zeros((len(image_filenames), label_size), np.float32)
        for idx in range(order.size):
            try:
                # load image
                img_filename = image_filenames[order[idx]]
                img = np.asarray(PIL.Image.open(img_filename))
                assert img.shape == (image_size, image_size, 3)
                img = PIL.Image.fromarray(img, 'RGB')
                # resize image
                img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                img = np.asarray(img)

                # load label, adjust params due to resize images
                labels[tfr.cur_images] = process_label_fn(img_filename)
                tfr.add_image(img)
            except:
                print(sys.exc_info()[1])
                continue
        tfr.add_labels(labels=labels[:tfr.cur_images, :])
        return tfr.cur_images


def create_300W_LP(tfrecord_dir, data_dir, print_progress, progress_interval):
    expected_images = {
        'AFW': 5207,
        'AFW_Flip': 5207,
        'HELEN': 37676,
        'HELEN_Flip': 5207,
        'IBUG': 1786,
        'IBUG_Flip': 1786,
        'LFPW': 16556,
        'LFPW_Flip': 16556
    }
    image_size = 450
    image_filenames = []
    for sub_name in os.walk(data_dir):
        print('processing %s' % sub_name)
        sub_folder = os.path.join(data_dir, sub_name)
        glob_pattern = os.path.join(sub_folder, '*.jpg')
        image_names = glob.glob(glob_pattern)
        if len(image_names) != expected_images[sub_name]:
            raise Exception('Expected to find %d images in \'%s\'' % (expected_images[sub_name], sub_folder))
        print('find %d images' % len(image_names))
        image_filenames.extend(image_names)

    return create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, image_size=image_size,
                           process_label_fn=extract_300W_LP_labels, print_progress=print_progress,
                           progress_interval=progress_interval)


def create_AFLW_2000(tfrecord_dir, data_dir, print_progress, progress_interval, bfm_path, resolution=224):
    expected_images = 2000
    image_size = 450

    sub_name = 'AFLW2000'
    print('processing %s' % sub_name)
    sub_folder = os.path.join(data_dir, sub_name)
    glob_pattern = os.path.join(sub_folder, '*.jpg')
    image_filenames = glob.glob(glob_pattern)
    if len(image_filenames) != expected_images:
        raise Exception('Expected to find %d images in \'%s\'' % (expected_images, sub_folder))
    print('find %d images' % len(image_filenames))
    # sort images
    image_filenames.sort()
    return create_tfrecord(tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, image_size=image_size,
                           process_label_fn=fn_extract_300W_LP_labels(bfm_path=bfm_path, image_size=image_size, is_aflw_2000=True), print_progress=print_progress,
                           progress_interval=progress_interval, resolution=resolution, random_shuffle=False)


if __name__ == '__main__':
    print_progress = True,
    progress_interval = 1000
    meta = Path('/opt/data/face-fuse/meta.json')
    tfrecord_dir = Path('/opt/data/face-fuse/test/')
    data_dir = Path('/opt/data')
    bfm_path = Path('/opt/data/BFM/BFM.mat')
    image_test_size = create_AFLW_2000(
        tfrecord_dir=tfrecord_dir,
        data_dir=data_dir,
        print_progress=print_progress,
        progress_interval=progress_interval,
        bfm_path=bfm_path
    )

    with open(meta, 'w') as f:
        json.dump(
            {
                'train_data_size': 0,
                'test_data_size': image_test_size
            },
            f
        )
