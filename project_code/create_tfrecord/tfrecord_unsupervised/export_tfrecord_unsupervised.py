import glob
import json
import os
import sys
from pathlib import Path

from project_code.create_tfrecord.export_tfrecord_util import load_image_from_file
from project_code.create_tfrecord.tfrecord_exporter import TFRecordExporterUnsupervised


def create_tfrecord(tfrecord_dir, image_filenames, mask_filenames, image_size, print_progress, progress_interval,
                    resolution=256, random_shuffle=True):
    with TFRecordExporterUnsupervised(tfrecord_dir=tfrecord_dir,
                                      expected_images=len(image_filenames),
                                      print_progress=print_progress,
                                      progress_interval=progress_interval) as tfr:
        order = tfr.choose_shuffled_order(random_shuffle=random_shuffle)

        for idx in range(order.size):
            try:
                # load image
                img = load_image_from_file(img_file=image_filenames[order[idx]], image_size=image_size,
                                           resolution=resolution, image_format='RGB')
                # load mask
                mask = load_image_from_file(img_file=mask_filenames[order[idx]], image_size=image_size,
                                            resolution=resolution, image_format='RGB')
                tfr.add_image(img, mask)
            except:
                print(sys.exc_info()[1])
                continue
        return tfr.cur_images


def create_dataset(tfrecord_dir, data_dir, expected_images, print_progress, progress_interval, resolution=256):
    image_size = 256
    image_filenames = []
    mask_filenames = []
    image_parent_folder = os.path.join(data_dir, 'nonlinear_face_3dmm/300W_LP/image')
    mask_parent_folder = os.path.join(data_dir, 'nonlinear_face_3dmm/300W_LP/mask_img')

    for dn in os.listdir(image_parent_folder):
        if dn not in expected_images:
            continue
        print('processing %s' % dn)
        image_folder = os.path.join(image_parent_folder, dn)
        glob_pattern = os.path.join(image_folder, '*png')
        image_files = glob.glob(glob_pattern)
        if len(image_files) != expected_images[dn]:
            raise Exception('Expected to find %d images in \'%s\', only found %d' % (
            expected_images[dn], image_folder, len(image_files)))
        print('find %d images' % len(image_files))
        image_filenames.extend(image_files)

        # get mask
        mask_folder = os.path.join(mask_parent_folder, dn)
        mask_names = [img_file.split('/')[-1] for img_file in image_files]
        mask_files = [os.path.join(mask_folder, mask_name) for mask_name in mask_names]
        if len(mask_files) != expected_images[dn]:
            raise Exception('Expected to find %d images in \'%s\', only found %d' % (
            expected_images[dn], mask_folder, len(mask_files)))
        print('find %d masks ' % len(mask_files))
        mask_filenames.extend(mask_files)

        # double check the order of image and mask are the same
        for image, mask in zip(image_filenames, mask_filenames):
            image_name = image.split('/')[-1]
            mask_name = mask.split('/')[-1]
            assert image_name == mask_name

    return create_tfrecord(
        tfrecord_dir=tfrecord_dir,
        image_filenames=image_filenames,
        mask_filenames=mask_filenames,
        image_size=image_size,
        print_progress=print_progress,
        progress_interval=progress_interval,
        resolution=resolution, random_shuffle=True
    )


def main(is_aflw, is_300w_lp, tfrecord_dir, data_dir, print_progress=True, progress_interval=1000):
    meta = Path(os.path.join(tfrecord_dir, 'meta.json'))
    tfrecord_train_dir = Path(os.path.join(tfrecord_dir, 'train'))
    tfrecord_test_dir = Path(os.path.join(tfrecord_dir, 'test'))

    if is_300w_lp:
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
        print('====== process 300W LP  =======')
        image_train_size = create_dataset(
            tfrecord_dir=tfrecord_train_dir,
            data_dir=data_dir,
            expected_images=expected_images,
            print_progress=print_progress,
            progress_interval=progress_interval,
            resolution=256 # we will crop the image randomly
        )
    else:
        image_train_size = 0

    if is_aflw:
        expected_images = {
            'AFLW2000': 2000,
        }
        print('====== process AFLW  =======')
        image_test_size = create_dataset(
            tfrecord_dir=tfrecord_test_dir,
            data_dir=data_dir,
            expected_images=expected_images,
            print_progress=print_progress,
            progress_interval=progress_interval,
            resolution=224 # for testing, we don't crop, so we need to rescale the image here
        )
    else:
        image_test_size = 0

    with open(meta, 'w') as f:
        json.dump(
            {
                'train_data_size': image_train_size,
                'test_data_size': image_test_size,
            },
            f
        )


if __name__ == '__main__':
    print_progress = True,
    progress_interval = 1000

    tfrecord_dir = Path('/opt/data/face-fuse/unsupervised')
    data_dir = Path('/opt/data')
    main(is_aflw=True, is_300w_lp=True, tfrecord_dir=tfrecord_dir, data_dir=data_dir)
