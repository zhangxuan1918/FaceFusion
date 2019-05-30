import os

import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

from project_code.data_tools.data_generator import get_3dmm_fine_tune_labeled_data

tf.random.set_seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

learning_rate = 0.0002
batch_size = 32
image_size = 224
epochs = 100

# load dataset
data_root_folder='H:/300W-LP/300W_LP/'
image_label_ds = get_3dmm_fine_tune_labeled_data(
    data_root_folder=data_root_folder
)

image_label_ds = image_label_ds.repeat()
image_label_ds = image_label_ds.batch(batch_size)
image_label_ds = image_label_ds.prefetch(buffer_size=AUTOTUNE)
print(image_label_ds)

for epoch in range(1):
    for images, labels in image_label_ds:
        # TODO
        # reshape labels to get individual params
        # for each parameter we set up an optimizer
        pass
