from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from absl import app
from absl import flags

from create_tfrecord import classifier_data_lib
from create_tfrecord.label_data_processor import LabelDataProcessor

FLAGS = flags.FLAGS

# BERT classification specific flags
flags.DEFINE_string(
    'input_data_dir', None,
    'The input data dir for the task.'
)

flags.DEFINE_enum('train_data_set_name', '300W_LP',
                  ['300W_LP'],
                  'training dataset name in input data dir.')

flags.DEFINE_enum('test_data_set_name', 'AFLW2000',
                  ['AFLW2000'],
                  'test dataset name in input data dir.')

# Shared flags across BERT fine-tuning tasks
flags.DEFINE_string(
    'train_data_output_path', None,
    'The path in which generated training input data will be written as tf records.'
)

flags.DEFINE_string(
    'test_data_output_path', None,
    'The path in which generated testing input data will be written as tf records.'
)

flags.DEFINE_string(
    'meta_data_file_path', None,
    'The path in which input meta data will be written.'
)


def generate_classifier_dataset():
    """Generates classifier dataset and returns input meta data"""
    assert FLAGS.input_data_dir and FLAGS.classification_task_name

    processor = LabelDataProcessor()

    return classifier_data_lib.generate_tf_record_from_data_file(
        processor,
        FLAGS.input_data_dir,
        FLAGS.train_data_set_name,
        FLAGS.test_data_set_name,
        train_data_output_path=FLAGS.train_data_output_path,
        test_data_output_path=FLAGS.test_data_output_path
    )


def main(_):
    input_meta_data = generate_classifier_dataset()

    with tf.io.gfile.GFile(FLAGS.meta_data_file_path, 'w') as writer:
        writer.write(json.dumps(input_meta_data, indent=4) + '\n')


if __name__ == '__main__':
    flags.mark_flag_as_required('input_data_dir')
    flags.mark_flag_as_required('train_data_set_name')
    flags.mark_flag_as_required('meta_data_file_path')
    app.run(main)
