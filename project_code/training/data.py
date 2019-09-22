from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from data_tools.data_generator import get_3dmm_warmup_data, get_3dmm_data
from morphable_model.model.morphable_model import FFTfMorphableModel


def setup_3dmm_warmup_data(
    bfm: FFTfMorphableModel,
    image_size_pre_shift: int,
    image_size: int,
    batch_size: int,
    data_train_dir: str,
    data_test_dir: str
):
    # load training dataset
    train_ds, test_ds = get_3dmm_warmup_data(
        bfm=bfm,
        im_size_pre_shift=image_size_pre_shift,
        im_size=image_size,
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir
    )

    # https://www.tensorflow.org/beta/guide/data
    train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds


def setup_3dmm_data(
    image_size: int,
    batch_size: int,
    data_train_dir: str,
    data_test_dir: str
):
    # load training dataset
    train_ds, test_ds = get_3dmm_data(
        im_size=image_size,
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir
    )

    train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds