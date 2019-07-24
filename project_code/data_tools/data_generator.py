import pathlib
import random
from functools import partial

import tensorflow as tf

from data_tools.data_util import load_3dmm_data_gen
from morphable_model.model.morphable_model import FFTfMorphableModel


def _get_3dmm_warmup_data_paths(folder, image_suffix):
    folder_pl = pathlib.Path(folder)
    image_paths = list(folder_pl.glob(image_suffix))
    image_paths = [str(p) for p in image_paths]
    gt_paths = [p.replace('.jpg', '.mat') for p in image_paths]

    return image_paths, gt_paths


def get_3dmm_warmup_data(
        bfm: FFTfMorphableModel,
        data_train_dir,
        data_test_dir
):
    train_image_paths, train_mat_paths = _get_3dmm_warmup_data_paths(folder=data_train_dir, image_suffix='*/*.jpg')
    random.shuffle(train_image_paths)
    print('3dmm warmup training data: {0}'.format(len(train_image_paths)))

    test_image_paths, test_mat_paths = _get_3dmm_warmup_data_paths(folder=data_test_dir, image_suffix='*.jpg')
    print('3dmm warmup testing data: {0}'.format(len(test_image_paths)))

    g_train_data = partial(load_3dmm_data_gen, bfm, '300W_LP', train_image_paths, train_mat_paths)
    train_ds = tf.data.Dataset.from_generator(
        g_train_data, output_types=(tf.float32, tf.string, {
            'shape': tf.float32,
            'pose': tf.float32,
            'exp': tf.float32,
            'color': tf.float32,
            'illum': tf.float32,
            'tex': tf.float32,
            'landmark': tf.float32
        }),
        output_shapes=(tf.TensorShape([224, 224, 3]), tf.TensorShape([]), {
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
        g_test_data,
        output_types=(tf.float32, {
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
