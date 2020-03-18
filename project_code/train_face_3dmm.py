import logging

from absl import app
from absl import flags
import tensorflow as tf


# REQUIRED PARAMS
from project_code.training.train_3dmm_supervised import TrainFaceModelSupervised
from project_code.training.train_3dmm_unsupervised import TrainFaceModelUnsupervised

flags.DEFINE_enum(
    'stage', None, ['SUPERVISED', 'UNSUPERVISED'],
    'One of {"SUPERVISED", "UNSUPERVISED"}. '
    '`supervised`: '
    'train the model with loss function, l2 norm between parameters '
    '`unsupervised`: '
    'train the model with loss function, l2 norm between rendered images'
)

flags.DEFINE_string(
    'model_dir', None,
    'model directory for saving trained model'
)

# PARMS WITH DEFAULT VALUE
flags.DEFINE_string(
    'bfm_dir', '/opt/data/BFM/',
    'BFM model directory, containing BFM.mat'
)

flags.DEFINE_string(
    'data_dir', '/opt/data/face-fuse/',
    'data directory for training and evaluating, containing train and test sub-folder'
)

flags.DEFINE_integer(
    'epochs', 3,
    'number of epochs for training'
)

flags.DEFINE_integer(
    'train_batch_size', 32,
    'batch size for training'
)

flags.DEFINE_integer(
    'eval_batch_size', 32,
    'batch size for evaluating'
)

flags.DEFINE_integer(
    'steps_per_loop', 10,
    'steps per loop, for efficiency. If steps_per_loop > 1, tf.function will be applied to training function '
    'to improve efficiency'
)

flags.DEFINE_float(
    'initial_lr', 0.0001,
    'initial learning rate, the learning rate will decay to 0 as training progress'
)

flags.DEFINE_string(
    'init_checkpoint', None,
    'initial checkpoint to restore model if provided, note if `init_checkpoint` is provided, ' 
    '`int_model_weight_path` will be ignored'
)

flags.DEFINE_string(
    'init_model_weight_path', None,
    'initial weights to load if provided, note if `init_checkpoint` is provided, ' 
    '`int_model_weight_path` will be ignored'
)

flags.DEFINE_integer(
    'resolution', 224,
    'input image resolution'
)

flags.DEFINE_integer(
    'num_gpu', 1,
    'number of gpus'
)

flags.DEFINE_enum(
    'backbone', 'resnet50', ['resnet50', 'resnet18'],
    'backbone to use, choose from [resnet50, resnet18]'
)

flags.DEFINE_enum(
    'distribute_strategy', 'one_device', ['one_device', 'mirrored'],
    'training distribution, choose from [one_device, mirrored]'
)

flags.DEFINE_bool(
    'run_eagerly', True,
    'run in eager mode, default true'
)

flags.DEFINE_integer(
    'model_output_size', 426,
    'number of face parameters, we remove region of interests, roi from the data'
)

flags.DEFINE_bool(
    'enable_profiler', False,
    'enable profiler, default false'
)

FLAGS = flags.FLAGS


def main(_):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info('Physical GPUs: %d, Logical GPUs: %d' % (len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.error(e)

    if FLAGS.stage == 'SUPERVISED':
        train_model = TrainFaceModelSupervised(
            data_dir=FLAGS.data_dir,  # data directory for training and evaluating
            model_dir=FLAGS.model_dir,
            # model directory for saving trained model
            epochs=FLAGS.epochs,  # number of epochs for training
            train_batch_size=FLAGS.train_batch_size,  # batch size for training
            eval_batch_size=FLAGS.eval_batch_size,  # batch size for evaluating
            steps_per_loop=FLAGS.steps_per_loop,  # steps per loop, for efficiency
            initial_lr=FLAGS.initial_lr,  # initial learning rate
            init_checkpoint=FLAGS.init_checkpoint,  # initial checkpoint to restore model if provided
            init_model_weight_path=FLAGS.init_model_weight_path,
            # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
            resolution=FLAGS.resolution,  # image resolution
            num_gpu=FLAGS.num_gpu,  # number of gpus
            stage=FLAGS.stage,  # stage name
            backbone=FLAGS.backbone,  # model architecture
            distribute_strategy=FLAGS.distribute_strategy,  # distribution strategy when num_gpu > 1
            run_eagerly=FLAGS.run_eagerly,
            model_output_size=FLAGS.model_output_size
        )
    elif FLAGS.stage == 'UNSUPERVISED':
        train_model = TrainFaceModelUnsupervised(
            bfm_dir=FLAGS.bfm_dir,
            data_dir=FLAGS.data_dir,  # data directory for training and evaluating
            model_dir=FLAGS.model_dir,
            # model directory for saving trained model
            epochs=FLAGS.epochs,  # number of epochs for training
            train_batch_size=FLAGS.train_batch_size,  # batch size for training
            eval_batch_size=FLAGS.eval_batch_size,  # batch size for evaluating
            steps_per_loop=FLAGS.steps_per_loop,  # steps per loop, for efficiency
            initial_lr=FLAGS.initial_lr,  # initial learning rate
            init_checkpoint=FLAGS.init_checkpoint,  # initial checkpoint to restore model if provided
            init_model_weight_path=FLAGS.init_model_weight_path,
            # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
            resolution=FLAGS.resolution,  # image resolution
            num_gpu=FLAGS.num_gpu,  # number of gpus
            stage=FLAGS.stage,  # stage name
            backbone=FLAGS.backbone,  # model architecture
            distribute_strategy=FLAGS.distribute_strategy,  # distribution strategy when num_gpu > 1
            run_eagerly=FLAGS.run_eagerly,
            model_output_size=FLAGS.model_output_size
        )
    else:
        raise ValueError('`stage` is invalid')

    train_model.train()


if __name__ == '__main__':
    flags.mark_flag_as_required('stage')
    flags.mark_flag_as_required('model_dir')

    app.run(main)
