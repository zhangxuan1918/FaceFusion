import logging
import os

from absl import app
from absl import flags
import tensorflow as tf

from project_code.training.dataset import TFRecordDataset
from project_code.training.model import init_model
from project_code.training.model_training_utils import run_customized_training_loop

flags.DEFINE_string('train_data_path', None, 'Path to training data for 3DMM')
flags.DEFINE_string('eval_data_path', None, 'Path to evaluating data for 3DMM')

flags.DEFINE_string(
    'input_meta_data_path', None, 'Path to file that contains meta dta about training and evaluating data'
)


def _run_3dmm_regressor_stage(
        strategy,
        input_meta_data,
        model=None,
        model_fn=None,
        model_dir=None,
        epochs=None,
        steps_per_epoch=None,
        steps_per_loop=None,
        eval_steps=None,
        initial_lr=None,
        init_checkpoint=None,
        model_weight_path=None,
        train_dir=None,
        eval_dir=None,
        run_early=None,
        resolution=None,
        batch_size=None,
        num_gpu=None
):
    # create training dataset
    train_dataset = TFRecordDataset(
        tfrecord_dir=train_dir,
        resolution=resolution,
        max_label_size='full',
        repeat=True,
        batch_size=batch_size,
        num_gpu=num_gpu
    )
    eval_dataset = TFRecordDataset(
        tfrecord_dir=eval_dir,
        resolution=resolution,
        max_label_size='full',
        repeat=False,
        batch_size=batch_size,
        num_gpu=num_gpu
    )

    # load model

    model = init_model(
        strategy=strategy,
        model=model,
        model_fn=model_fn,
        opt_fn=opt_fn,
        initial_lr=initial_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        init_checkpoint=init_checkpoint,
        model_weight_path=model_weight_path)

    model = run_customized_training_loop(
        strategy=strategy,
        model=model,
        loss_fn=loss_fn,
        model_dir=model_dir,
        train_dataset=train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        steps_per_loop=steps_per_loop,
        eval_dataset=eval_dataset,
        run_eagerly=run_early
    )
    return model


def run_3dmm_regressor(
        strategy_name,
        input_meta_data,
        model_dir,
        epochs,
        steps_per_epoch,
        steps_per_loop,
        eval_steps,
        initial_lr,
        init_checkpoint=None,
        model_weight_path=None,
        train_dir,
        eval_dir,
        run_early=False,
        resolution,
        batch_size,
        num_gpu,
        is_run_supervised,
        is_run_unsupervised,
):
    output_size = input_meta_data['output_size']

    if strategy_name.lower() == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

    model = None
    if is_run_supervised:
        model_dir_unsupervised = os.path.join(model_dir, 'supervised')
        model = _run_3dmm_regressor_stage(
            strategy=strategy,
            input_meta_data=input_meta_data,
            model=model,
            model_fn=model_fn,
            model_dir=model_dir_unsupervised,
            epochs=epochs_supervised,
            steps_per_epoch=None,
            steps_per_loop=None,
            eval_steps=None,
            initial_lr=None,
            init_checkpoint=None,
            model_weight_path=None,
            train_dir=None,
            eval_dir=None,
            run_early=None,
            resolution=None,
            batch_size=None,
            num_gpu=None
        )

    if is_run_unsupervised:
        model_dir_unsupervised = os.path.join(model_dir, 'unsupervised')
        model = _run_3dmm_regressor_stage(
            strategy=strategy,
            input_meta_data=input_meta_data,
            model=model,
            model_fn=None,
            model_dir=model_dir_unsupervised,
            epochs=epochs_unsupervised,
            steps_per_epoch=None,
            steps_per_loop=None,
            eval_steps=None,
            initial_lr=None,
            init_checkpoint=None,
            model_weight_path=None,
            train_dir=None,
            eval_dir=None,
            run_early=None,
            resolution=None,
            batch_size=None,
            num_gpu=None
        )
