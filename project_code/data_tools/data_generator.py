import pathlib
import random
from functools import partial

import tensorflow as tf

from project_code.data_tools.data_util import load_3dmm_data


def _get_3dmm_warmup_data_paths(folder, image_suffix):
    folder_pl = pathlib.Path(folder)
    image_paths = list(folder_pl.glob(image_suffix))
    image_paths = [str(p) for p in image_paths]
    gt_paths = [p.replace('.jpg', '.mat') for p in image_paths]

    return image_paths, gt_paths


def get_3dmm_warmup_data(
        data_train_dir,
        data_test_dir,
        image_suffix
):
    train_image_paths, train_mat_paths = _get_3dmm_warmup_data_paths(folder=data_train_dir, image_suffix=image_suffix)
    random.shuffle(train_image_paths)
    print('3dmm warmup training data: {0}'.format(len(train_image_paths)))

    test_image_paths, test_mat_paths = _get_3dmm_warmup_data_paths(folder=data_test_dir, image_suffix=image_suffix)
    print('3dmm warmup testing data: {0}'.format(len(test_image_paths)))

    train_paths_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mat_paths))
    fn_train_data = partial(load_3dmm_data, '300W_LP')
    train_ds = train_paths_ds.map(fn_train_data)

    test_paths_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_mat_paths))
    fn_test_data = partial(load_3dmm_data, 'AFLW_2000')
    test_ds = test_paths_ds.map(fn_test_data)

    return train_ds, test_ds
