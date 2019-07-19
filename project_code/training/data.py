from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_warmup_data


def setup_3dmm_warmup_data(
    batch_size,
    data_train_dir,
    data_test_dir,
    image_suffix='*/*.jpg'
):
    # load training dataset
    train_ds, test_ds = get_3dmm_warmup_data(
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir,
        image_suffix=image_suffix,
    )

    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds


def setup_3dmm_data():
    return None, None