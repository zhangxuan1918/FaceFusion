import glob
import json
import os
import sys
from pathlib import Path
import random

import numpy as np

from project_code.create_tfrecord.tfrecord_exporter import TFRecordExporterSupervised
from project_code.create_tfrecord.export_tfrecord_util import load_image_from_file, fn_extract_ffhq_labels


def create_tfrecord(tfrecord_dir, image_filenames, image_size, process_label_fn, print_progress, progress_interval,
                    resolution=224, label_size=138, random_shuffle=True):
    with TFRecordExporterSupervised(tfrecord_dir=tfrecord_dir,
                                    expected_images=len(image_filenames),
                                    print_progress=print_progress,
                                    progress_interval=progress_interval) as tfr:
        order = tfr.choose_shuffled_order(random_shuffle=random_shuffle)
        labels = np.zeros((len(image_filenames), label_size), np.float32)
        for idx in range(order.size):
            try:
                # load image
                img = load_image_from_file(img_file=image_filenames[order[idx]], image_size=image_size,
                                           resolution=resolution, image_format='RGB')

                # load label, adjust params due to resize images
                labels[tfr.cur_images] = process_label_fn(image_filenames[order[idx]])
                tfr.add_image(img)
            except Exception as e:
                print(e)
                continue
        tfr.add_labels(labels=labels[:tfr.cur_images, :])
        return tfr.cur_images


def create_ffhq(tfrecord_dir, files, print_progress, progress_interval, txt_suffix, label_size, resolution=224):
    image_size = 1024

    return create_tfrecord(
        tfrecord_dir=tfrecord_dir,
        image_filenames=files,
        image_size=image_size,
        process_label_fn=fn_extract_ffhq_labels(image_size=image_size, txt_suffix=txt_suffix),
        print_progress=print_progress,
        progress_interval=progress_interval,
        label_size=label_size,
        resolution=resolution, random_shuffle=True
    )


def main(tfrecord_dir, data_dir, label_size=185, print_progress=True, progress_interval=1000,
         test_data_size=2000, txt_suffix='_2d_lms.txt'):
    meta = Path(os.path.join(tfrecord_dir, 'meta.json'))
    tfrecord_train_dir = Path(os.path.join(tfrecord_dir, 'train'))
    tfrecord_test_dir = Path(os.path.join(tfrecord_dir, 'test'))
    image_filenames = []
    sub_name = 'ffhq-dataset'
    sub_folder = os.path.join(data_dir, sub_name)
    for img_name in os.listdir(sub_folder):
        if img_name.endswith('png'):
            img_file = os.path.join(sub_folder, img_name)
            image_filenames.append(img_file)

    print('total file %d' % len(image_filenames))
    random.shuffle(image_filenames)
    train_filenames = image_filenames[:-test_data_size]
    test_filenames = image_filenames[-test_data_size:]
    print('training data size: %d' % len(train_filenames))
    print('testing data size: %d' % len(test_filenames))
    print('====== process face-ffhq  =======')

    image_train_size = create_ffhq(
        tfrecord_dir=tfrecord_train_dir,
        files=train_filenames,
        print_progress=print_progress,
        progress_interval=progress_interval,
        txt_suffix=txt_suffix,
        label_size=label_size
    )

    image_test_size = create_ffhq(
        tfrecord_dir=tfrecord_test_dir,
        files=test_filenames,
        print_progress=print_progress,
        progress_interval=progress_interval,
        txt_suffix=txt_suffix,
        label_size=label_size
    )

    with open(meta, 'w') as f:
        json.dump(
            {
                'train_data_size': image_train_size,
                'test_data_size': image_test_size,
                'output_size': label_size
            },
            f
        )


if __name__ == '__main__':
    print_progress = True,
    progress_interval = 1000
    label_size = 136
    test_data_size = 2000

    tfrecord_dir = Path('/opt/data/face-fuse/supervised_ffhq')
    data_dir = Path('/opt/data')
    main(tfrecord_dir=tfrecord_dir, data_dir=data_dir, label_size=label_size)
