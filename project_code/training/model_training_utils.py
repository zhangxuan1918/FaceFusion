import json
import logging
import os

import tensorflow as tf


def _save_checkpoint(checkpoint, model_dir, checkpoint_prefix):
    checkpoint_path = os.path.join(model_dir, checkpoint_prefix)
    saved_path = checkpoint.save(checkpoint_path)
    logging.info('Saving model as TF checkpoint: %s', saved_path)
    return


def _float_metric_value(metric):
    return metric.reset().numpy().astype(float)


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


def run_customized_training_loop(
        _sentinel=None,
        strategy=None,
        model=None,
        loss_fn=None,
        model_dir=None,
        train_dataset=None,
        epochs=1,
        steps_per_epoch=None,
        steps_per_loop=1,
        eval_dataset=None,
        run_eagerly=False
):
    if _sentinel is not None:
        raise ValueError('Only call `run_customized_training_loop()` ' 
                         'with named arguments')

    required_arguments = [model, loss_fn, model_dir, train_dataset, steps_per_epoch, steps_per_loop]

    if any([arg is None for arg in required_arguments]):
        raise ValueError('`loss_fn`, `model_dir`, `train_dataset` are required parameters.')

    assert tf.executing_eagerly()

    total_training_steps = steps_per_epoch * epochs

    optimizer = model.optimizer
    use_float16 = isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

    summary_dir = os.path.join(model_dir, 'summaries')
    eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'eval'))
    training_vars = model.trainable_variables

    def _replicated_step(inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            model_outputs = model(inputs, training=True)
            loss = loss_fn(labels, model_outputs)
            if use_float16:
                scaled_loss = optimizer.get_scaled_loss(loss)
        if use_float16:
            scaled_grads = tape.gradient(scaled_loss, training_vars)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, training_vars)
        optimizer.apply_gradients(zip(grads, training_vars))

        train_loss_metric.update_state(loss)

    @tf.function
    def train_steps(iterator, steps):
        if not isinstance(steps, tf.Tensor):
            raise ValueError('`steps` should be an Tensor. Python objects can cause retracing')

        for _ in tf.range(steps):
            strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

    def train_single_step(iterator):
        strategy.experimental_run_v2(_replicated_step, args=(next(iterator),))

    def test_step(iterator):

        def _test_step_fn(inputs):
            inputs, labels = inputs
            model_outputs = model(inputs, training=False)

        strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

    if not run_eagerly:
        train_single_step = tf.function(train_single_step)
        test_step = tf.function(test_step)

    def _run_evaluation(current_training_step, test_iterator):
        while True:
            try:
                test_step(test_iterator)
            except tf.errors.OutOfRangeError:
                break

        with eval_summary_writer.as_default():
            eval_loss = _float_metric_value(eval_loss_metric)
            logging.info('Step: [%d] Validation %s = %f', current_training_step,
                         eval_loss_metric.name, eval_loss)
            eval_summary_writer.flush()

        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if latest_checkpoint_file:
            logging.info('Checkpoint file %s found and restoring from checkpoint', latest_checkpoint_file)
            checkpoint.restore(latest_checkpoint_file)
            logging.info('Loading from checkpoint file completed')
        current_step = optimizer.iterations.numpy()
        checkpoint_name = 'rtl_step_{step}.ckpt'

        while current_step < total_training_steps:
            train_loss_metric.reset_states()

            steps = steps_to_run(current_step, steps_per_epoch, steps_per_loop)

            if steps == 1:
                train_single_step(train_dataset)
            else:
                train_steps(train_dataset, tf.convert_to_tensor(steps, dtype=tf.int32))

            current_step += steps

            train_loss = _float_metric_value(train_loss_metric)
            training_status = 'Train Step: %d/%d  / loss = %s' % (
                current_step, total_training_steps, train_loss)

            logging.info(training_status)

            if current_step % steps_per_epoch == 0:
                if current_step < total_training_steps:
                    _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))

                if eval_dataset:
                    logging.info('Running evaluation after step: %s', current_step)
                    eval_dataset.reset_iterator()
                    _run_evaluation(current_step, eval_dataset)
                    eval_loss_metric.reset_states()

        _save_checkpoint(checkpoint, model_dir, checkpoint_name.format(step=current_step))

        if eval_dataset:
            logging.info('Running final evaluation after training is complete.')
            eval_dataset.reset_iterator()
            _run_evaluation(current_step, eval_dataset)

        training_summary = {
            'total_training_steps': total_training_steps,
            'train_loss': _float_metric_value(train_loss_metric)
        }

        if eval_dataset:
            training_summary['eval_metric'] = _float_metric_value(eval_loss_metric)

        write_txt_summary(training_summary, summary_dir)

        return model