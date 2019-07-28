import pathlib
import random
from functools import partial

import tensorflow as tf

from data_tools.data_util import load_3dmm_data_gen, load_image_3dmm
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
    return zip(*combined)


def get_3dmm_warmup_data(
        bfm: FFTfMorphableModel,
        data_train_dir,
        data_test_dir
):
    train_image_paths, train_mat_paths = _get_3dmm_warmup_data_paths(folder=data_train_dir, image_suffix='*/*.jpg')
    print('3dmm warmup training data: {0}'.format(len(train_image_paths)))

    test_image_paths, test_mat_paths = _get_3dmm_warmup_data_paths(folder=data_test_dir, image_suffix='*.jpg')
    print('3dmm warmup testing data: {0}'.format(len(test_image_paths)))

    g_train_data = partial(load_3dmm_data_gen, bfm, '300W_LP', train_image_paths, train_mat_paths)
    train_ds = tf.data.Dataset.from_generator(
        g_train_data, output_types=(tf.float32, {
            'shape': tf.float32,
            'pose': tf.float32,
            'exp': tf.float32,
            'color': tf.float32,
            'illum': tf.float32,
            'tex': tf.float32,
            'landmark': tf.float32
        }),
        output_shapes=(tf.TensorShape([224, 224, 3]), {
            'shape': tf.TensorShape([199, 1]),
            'pose': tf.TensorShape([1, 7]),
            'exp': tf.TensorShape([29, 1]),
            'color': tf.TensorShape([1, 7]),
            'illum': tf.TensorShape([1, 10]),
            'tex': tf.TensorShape([199, 1]),
            'landmark': tf.TensorShape([2, 68])
        })
    )

    g_test_data = partial(load_3dmm_data_gen, bfm, 'AFLW_2000', test_image_paths, test_mat_paths)
    test_ds = tf.data.Dataset.from_generator(
        g_test_data, output_types=(tf.float32, {
            'shape': tf.float32,
            'pose': tf.float32,
            'exp': tf.float32,
            'color': tf.float32,
            'illum': tf.float32,
            'tex': tf.float32,
            'landmark': tf.float32
        }),
        output_shapes=(tf.TensorShape([224, 224, 3]), {
            'shape': tf.TensorShape([199, 1]),
            'pose': tf.TensorShape([1, 7]),
            'exp': tf.TensorShape([29, 1]),
            'color': tf.TensorShape([1, 7]),
            'illum': tf.TensorShape([1, 10]),
            'tex': tf.TensorShape([199, 1]),
            'landmark': tf.TensorShape([2, 68])
        })
    )

    return train_ds, test_ds


def get_3dmm_data(
    data_train_dir,
    data_test_dir
):
    train_image_paths = _get_files(folder=data_train_dir, file_pattern='*.png')
    random.shuffle(train_image_paths)
    print('3dmm training data: {0}'.format(len(train_image_paths)))

    test_image_paths = _get_files(folder=data_test_dir, file_pattern='*.png')
    print('3dmm testing data: {0}'.format(len(test_image_paths)))

    train_ds = tf.data.Dataset.from_tensor_slices(train_image_paths).map(load_image_3dmm)
    test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths).map(load_image_3dmm)

    return train_ds, test_ds
