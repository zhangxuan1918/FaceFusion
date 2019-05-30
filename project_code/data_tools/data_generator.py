import pathlib
import random

import tensorflow as tf

from project_code.data_tools.data_util import load_image_labels_3dmm

data_root_folder = ''
data_root = pathlib.Path(data_root_folder)
all_image_paths = list(data_root.glob('*/*.jpg'))
all_image_paths = [str(p) for p in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print('num of images: {0}'.format(image_count))

all_labels_paths = [p.replace('/.jpg', '/.mat') for p in all_image_paths]
print('num of labels: {0}'.format(all_labels_paths))

image_label_paths_ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels_paths))
image_label_ds = image_label_paths_ds.map(load_image_labels_3dmm)
print(image_label_ds)