import glob
import json
import os
import sys
from pathlib import Path
import random

import numpy as np

from project_code.create_tfrecord.tfrecord_exporter import TFRecordExporterSupervised
from project_code.create_tfrecord.export_tfrecord_util import fn_extract_80k_labels, load_image_from_file


def create_tfrecord(tfrecord_dir, image_filenames, image_size, process_label_fn, print_progress, progress_interval,
                    resolution=224, label_size=185, random_shuffle=True):
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
            except:
                print(sys.exc_info()[1])
                continue
        tfr.add_labels(labels=labels[:tfr.cur_images, :])
        return tfr.cur_images


def create_80k(tfrecord_dir, files, print_progress, progress_interval, param_mean_std_path, label_size, resolution=224):
    image_size = 450

    return create_tfrecord(
        tfrecord_dir=tfrecord_dir,
        image_filenames=files,
        image_size=image_size,
        process_label_fn=fn_extract_80k_labels(param_mean_std_path=param_mean_std_path, image_size=image_size),
        print_progress=print_progress,
        progress_interval=progress_interval,
        label_size=label_size,
        resolution=resolution, random_shuffle=True
    )


def main(tfrecord_dir, data_dir, param_mean_std_path, label_size=185, print_progress=True, progress_interval=1000,
         test_data_size=2000):
    meta = Path(os.path.join(tfrecord_dir, 'meta.json'))
    tfrecord_train_dir = Path(os.path.join(tfrecord_dir, 'train'))
    tfrecord_test_dir = Path(os.path.join(tfrecord_dir, 'test'))
    image_filenames = []
    sub_name = 'face-80k/Coarse_Dataset/CoarseData'
    sub_folder = os.path.join(data_dir, sub_name)
    for dn in os.listdir(sub_folder):
        print('processing %s' % dn)
        im_folder = os.path.join(sub_folder, dn)
        glob_pattern = os.path.join(im_folder, '*.jpg')
        image_names = glob.glob(glob_pattern)
        print('find %d images' % len(image_names))
        image_filenames.extend(image_names)

    print('total file %d' % len(image_filenames))
    random.shuffle(image_filenames)
    train_filenames = image_filenames[:-test_data_size]
    test_filenames = image_filenames[-test_data_size:]
    print('training data size: %d' % len(train_filenames))
    print('testing data size: %d' % len(test_filenames))
    print('====== process face-80k  =======')

    image_train_size = create_80k(
        tfrecord_dir=tfrecord_train_dir,
        files=train_filenames,
        print_progress=print_progress,
        progress_interval=progress_interval,
        param_mean_std_path=param_mean_std_path,
        label_size=label_size
    )

    image_test_size = create_80k(
        tfrecord_dir=tfrecord_test_dir,
        files=test_filenames,
        print_progress=print_progress,
        progress_interval=progress_interval,
        param_mean_std_path=param_mean_std_path,
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
    label_size = 185
    test_data_size = 2000

    tfrecord_dir = Path('/opt/data/face-fuse/supervised_80k')
    data_dir = Path('/opt/data')
    param_mean_std_path = Path('/opt/data/face-fuse/stats_80k.npz')
    main(tfrecord_dir=tfrecord_dir, data_dir=data_dir, param_mean_std_path=param_mean_std_path, label_size=label_size)
