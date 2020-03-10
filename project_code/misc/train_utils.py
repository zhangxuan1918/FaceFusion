import json
import logging
import os

import tensorflow as tf


def save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    logging.info('Saving model as TF checkpoint: %s', saved_path)
    return


def float_metric_value(metric):
    return metric.result().numpy().astype(float)


def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    if steps_per_loop <= 0:
        raise ValueError('`steps_per_loop` should be positive integer')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
        return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop


def write_txt_summary(training_summary, summary_dir):
    summary_path = os.path.join(summary_dir, 'training_summary.txt')

    with tf.io.gfile.GFile(summary_path, 'wb') as f:
        logging.info('Training Summary: \n%s', str(training_summary))
        f.write(json.dumps(training_summary, indent=4))
