import logging
import tensorflow as tf

from project_code.misc import distribution_utils


def init_model(strategy, model, model_fn, opt_fn, initial_lr, steps_per_epoch, epochs, init_checkpoint, model_weight_path):
    if model is None and model_fn is None:
        raise ValueError('`model` and `model_fn` cannot be None at the same time')
    with distribution_utils.get_strategy_scope(strategy):
        # To correctly place the model weights on accelerators,
        # model and optimizer should be created in scope.
        if model is None:
            model = model_fn()

        model.optimizer = opt_fn(
            initial_lr, steps_per_epoch * epochs
        )

        if init_checkpoint:
            logging.info(
                'Checkpoint file %s found and restoring from '
                'initial checkpoint for core model.', init_checkpoint)
            checkpoint = tf.train.Checkpoint(model=sub_model)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
            logging.info('Loading from checkpoint file completed')
        elif model_weight_path:
            # TODO load weights
            pass
    return model
