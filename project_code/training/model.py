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
