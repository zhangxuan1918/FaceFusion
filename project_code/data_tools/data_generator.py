import pathlib
import random
from functools import partial

import numpy as np
import tensorflow as tf

from data_tools.data_util import load_image_3dmm, load_3dmm_data
from morphable_model.model.morphable_model import FFTfMorphableModel


def _get_files(folder, file_pattern):
    folder = pathlib.Path(folder)
    files = list(folder.glob(file_pattern))
    files = [str(p) for p in files]
    return files


def _get_3dmm_warmup_data_paths(folder, image_suffix):
    image_paths = _get_files(folder=folder, file_pattern=image_suffix)
    gt_paths = [p.replace('.jpg', '.mat') for p in image_paths]

    # shuffle data
    combined = list(zip(image_paths, gt_paths))
    random.shuffle(combined)
    x, y = zip(*combined)
    return list(x), list(y)


def get_3dmm_warmup_data(
        bfm: FFTfMorphableModel,
        im_size_pre_shift: int,
        im_size: int,
        data_train_dir: str,
        data_test_dir: str
):
    train_image_paths, train_mat_paths = _get_3dmm_warmup_data_paths(folder=data_train_dir, image_suffix='*/*.jpg')
    print('3dmm warmup training data: {0}'.format(len(train_image_paths)))

    test_image_paths, test_mat_paths = _get_3dmm_warmup_data_paths(folder=data_test_dir, image_suffix='*.jpg')
    print('3dmm warmup testing data: {0}'.format(len(test_image_paths)))

    g_train_data = partial(load_3dmm_data, bfm, im_size_pre_shift, im_size, '300W_LP')

    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mat_paths)).map(g_train_data)

    g_test_data = partial(load_3dmm_data, bfm, im_size_pre_shift, im_size, 'AFLW_2000')
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_mat_paths)).map(g_test_data)

    return train_ds, test_ds


def get_3dmm_data(
        im_size: int,
        data_train_dir: str,
        data_test_dir: str
):
    train_image_paths = _get_files(folder=data_train_dir, file_pattern='*.png')
    random.shuffle(train_image_paths)
    print('3dmm training data: {0}'.format(len(train_image_paths)))

    test_image_paths = _get_files(folder=data_test_dir, file_pattern='*.png')
    print('3dmm testing data: {0}'.format(len(test_image_paths)))

    txys = np.zeros(shape=len(train_image_paths))
    g_load_image_data = partial(load_image_3dmm, im_size, im_size, txys, txys)

    train_ds = tf.data.Dataset.from_tensor_slices(train_image_paths).map(g_load_image_data)
    test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths).map(g_load_image_data)

    return train_ds, test_ds
