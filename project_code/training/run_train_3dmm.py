import logging
import math
import os

from absl import app
from absl import flags
import tensorflow as tf

from project_code.misc.utils import EasyDict
from project_code.training.dataset import TFRecordDataset

flags.DEFINE_string('train_data_path', None, 'Path to training data for 3DMM')
flags.DEFINE_string('eval_data_path', None, 'Path to evaluating data for 3DMM')

flags.DEFINE_string(
    'input_meta_data_path', None, 'Path to file that contains meta dta about training and evaluating data'
)


def config(input_meta_data, data_dir, model_dir, stage):
    config = EasyDict()
    if stage == 'supervised':
        config.output_size = input_meta_data['num_output']
        config.epochs = 1
        config.train_batch_size = 16
        config.eval_batch_size = 16
        config.steps_per_loop = 1
        config.steps_per_epoch = int(input_meta_data['train_data_size'] / config.train_batch_size)
        config.eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / config.eval_batch_size))
        config.initial_lr = 0.001
        config.init_checkpoint = None
        config.model_weight_path = None
        config.train_dir = os.path.join(data_dir, 'train')
        config.eval_dir = os.path.join(data_dir, 'test')
        config.resolution = 224
        config.num_gpu = 1
        return config
    elif stage == 'unsupervised':
        config.output_size = input_meta_data['num_output']
        config.epochs = 2
        config.train_batch_size = 16
        config.eval_batch_size = 16
        config.steps_per_loop = 1
        config.steps_per_epoch = int(input_meta_data['train_data_size'] / config.train_batch_size)
        config.eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / config.eval_batch_size))
        config.initial_lr = 0.0005
        config.init_checkpoint = None
        config.model_weight_path = None
        config.train_dir = '/opt/data/face-fuse/train/'
        config.eval_dir = '/opt/data/face-fuse/test/'
        config.model_dir = os.path.join(model_dir, 'unsupervised')
        config.resolution = 224
        config.num_gpu = 1
        return config
    else:
        raise ValueError('`stage` not supported: %s' % stage)


def _run_3dmm_regressor_stage(
        strategy,
        model,
        config,
        run_eagerly
):
    # create training dataset
    train_dataset = TFRecordDataset(
        tfrecord_dir=config.train_dir,
        resolution=config.resolution,
        max_label_size='full',
        repeat=True,
        batch_size=config.train_batch_size,
        num_gpu=config.num_gpu
    )
    eval_dataset = TFRecordDataset(
        tfrecord_dir=config.eval_dir,
        resolution=config.resolution,
        max_label_size='full',
        repeat=False,
        batch_size=config.eval_batch_size,
        num_gpu=config.num_gpu
    )

    # load model

    model = init_model(
        strategy=strategy,
        model=model,
        model_fn=config.model_fn,
        opt_fn=config.opt_fn,
        init_checkpoint=config.init_checkpoint,
        model_weight_path=config.model_weight_path)

    model = run_customized_training_loop(
        strategy=strategy,
        model=model,
        loss_fn=config.loss_fn,
        model_dir=config.model_dir,
        train_dataset=train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        steps_per_loop=config.steps_per_loop,
        eval_dataset=eval_dataset,
        run_eagerly=run_eagerly
    )
    return model


def run_3dmm_regressor(
        strategy_name,
        config_supervised,
        config_unpervised,
        is_run_supervised,
        is_run_unsupervised,
        run_eagerly=True
):

    if strategy_name.lower() == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    model = None
    if is_run_supervised:
        model = _run_3dmm_regressor_stage(
            strategy=strategy,
            model=model,
            config=config_supervised,
            run_eagerly=run_eagerly
        )

    if is_run_unsupervised:
        model = _run_3dmm_regressor_stage(
            strategy=strategy,
            model=model,
            config=config_unpervised,
            run_eagerly=run_eagerly
        )
