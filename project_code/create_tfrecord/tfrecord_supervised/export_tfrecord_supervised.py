import glob
import json
import os
import sys
from pathlib import Path

import PIL
import numpy as np

from project_code.create_tfrecord.tfrecord_exporter import TFRecordExporterSupervised
from project_code.create_tfrecord.export_tfrecord_util import fn_extract_300W_LP_labels, load_image_from_file


def create_tfrecord(tfrecord_dir, image_filenames, image_size, process_label_fn, print_progress, progress_interval,
                    resolution=224, label_size=430, random_shuffle=True):
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


def create_300W_LP(tfrecord_dir, data_dir, print_progress, progress_interval, param_mean_std_path, label_size, resolution=224):
    expected_images = {
        'AFW': 5207,
        'AFW_Flip': 5207,
        'HELEN': 37676,
        'HELEN_Flip': 37676,
        'IBUG': 1786,
        'IBUG_Flip': 1786,
        'LFPW': 16556,
        'LFPW_Flip': 16556
    }
    image_size = 450
    image_filenames = []
    sub_name = '300W_LP'
    sub_folder = os.path.join(data_dir, sub_name)
    for dn in os.listdir(sub_folder):
        if dn not in expected_images:
            continue
        print('processing %s' % dn)
        im_folder = os.path.join(sub_folder, dn)
        glob_pattern = os.path.join(im_folder, '*.jpg')
        image_names = glob.glob(glob_pattern)
        if len(image_names) != expected_images[dn]:
            raise Exception('Expected to find %d images in \'%s\', only found %d' % (expected_images[dn], sub_folder, len(image_names)))
        print('find %d images' % len(image_names))
        image_filenames.extend(image_names)

    return create_tfrecord(
        tfrecord_dir=tfrecord_dir,
        image_filenames=image_filenames,
        image_size=image_size,
        process_label_fn=fn_extract_300W_LP_labels(param_mean_std_path=param_mean_std_path, image_size=image_size, is_aflw_2000=False),
        print_progress=print_progress,
        progress_interval=progress_interval,
        label_size=label_size,
        resolution=resolution, random_shuffle=True
    )


def create_AFLW_2000(tfrecord_dir, data_dir, print_progress, progress_interval, param_mean_std_path, label_size, resolution=224):
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
    return create_tfrecord(
        tfrecord_dir=tfrecord_dir, image_filenames=image_filenames, image_size=image_size,
        process_label_fn=fn_extract_300W_LP_labels(param_mean_std_path=param_mean_std_path, image_size=image_size, is_aflw_2000=True),
        print_progress=print_progress,
        progress_interval=progress_interval,
        label_size=label_size,
        resolution=resolution, random_shuffle=False)


def main(is_aflw, is_300w_lp, tfrecord_dir, data_dir, param_mean_std_path, label_size=430, print_progress=True, progress_interval=1000):
    meta = Path(os.path.join(tfrecord_dir, 'meta.json'))
    tfrecord_train_dir = Path(os.path.join(tfrecord_dir, 'train'))
    tfrecord_test_dir = Path(os.path.join(tfrecord_dir, 'test'))

    if is_300w_lp:
        print('====== process 300W LP  =======')
        image_train_size = create_300W_LP(
            tfrecord_dir=tfrecord_train_dir,
            data_dir=data_dir,
            print_progress=print_progress,
            progress_interval=progress_interval,
            param_mean_std_path=param_mean_std_path,
            label_size=label_size
        )
    else:
        image_train_size = 0

    if is_aflw:
        print('====== process AFLW  =======')
        image_test_size = create_AFLW_2000(
            tfrecord_dir=tfrecord_test_dir,
            data_dir=data_dir,
            print_progress=print_progress,
            progress_interval=progress_interval,
            param_mean_std_path=param_mean_std_path,
            label_size=label_size
        )
    else:
        image_test_size = 0

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
    label_size = 430

    tfrecord_dir = Path('/opt/data/face-fuse/supervised')
    data_dir = Path('/opt/data')
    param_mean_std_path = Path('/opt/data/face-fuse/stats_300W_LP.npz')
    main(is_aflw=True, is_300w_lp=True, tfrecord_dir=tfrecord_dir, data_dir=data_dir, param_mean_std_path=param_mean_std_path,
         label_size=label_size)
