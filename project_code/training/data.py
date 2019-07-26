from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from data_tools.data_generator import get_3dmm_warmup_data, get_3dmm_data
from morphable_model.model.morphable_model import FFTfMorphableModel


def setup_3dmm_warmup_data(
    bfm: FFTfMorphableModel,
    batch_size,
    data_train_dir,
    data_test_dir
):
    # load training dataset
    train_ds, test_ds = get_3dmm_warmup_data(
        bfm=bfm,
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir
    )

    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds


def setup_3dmm_data(
    batch_size,
    data_train_dir,
    data_test_dir
):
    # load training dataset
    train_ds, test_ds = get_3dmm_data(
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir
    )

    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds