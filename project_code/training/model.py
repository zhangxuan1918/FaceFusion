import logging
import tensorflow as tf

from project_code.misc import distribution_utils
from project_code.models.resnet18 import Resnet18
from project_code.models.resnet50 import Resnet50


def get_model_fn(image_size, output_size, model_type):
    if model_type not in ['resnet50', 'resnet18']:
        raise ValueError('`model_type` not supported: %s' % model_type)

    def regression_model():
        if model_type == 'resnet50':
            return Resnet50(image_size=image_size, output_size=output_size)
        else:
            return Resnet18(image_size=image_size, output_size=output_size)
    return regression_model


def init_model(strategy, model, model_fn, opt_fn, init_checkpoint, model_weight_path):
    if model is None and model_fn is None:
        raise ValueError('`model` and `model_fn` cannot be None at the same time')
    with distribution_utils.get_strategy_scope(strategy):
        # To correctly place the model weights on accelerators,
        # model and optimizer should be created in scope.
        if model is None:
            model = model_fn()

        model.optimizer = opt_fn()

        if init_checkpoint:
            logging.info(
                'Checkpoint file %s found and restoring from '
                'initial checkpoint for core model.', init_checkpoint)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
            logging.info('Loading from checkpoint file completed')
        elif model_weight_path:
            # TODO load weights
            pass
    return model
