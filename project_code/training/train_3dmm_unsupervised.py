import json
import logging
import math
import os

import tensorflow as tf
from absl import flags
from tf_3dmm.mesh.reader import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, \
    unnormalize_labels
from project_code.misc import distribution_utils
from project_code.misc.image_utils import process_reals
from project_code.misc.train_utils import float_metric_value, steps_to_run, save_checkpoint, write_txt_summary
from project_code.models.resnet18 import Resnet18
from project_code.models.resnet50 import Resnet50
from project_code.training.dataset import TFRecordDataset
from project_code.training.optimization import AdamWeightDecay

logging.basicConfig(level=logging.INFO)
flags.DEFINE_string('train_data_path', None, 'Path to training data for 3DMM')
flags.DEFINE_string('eval_data_path', None, 'Path to evaluating data for 3DMM')

flags.DEFINE_string(
    'input_meta_data_path', None, 'Path to file that contains meta dta about training and evaluating data'
)


class TrainFaceModelUnsupervised:

    def __init__(self,
                 bfm_dir, # face model directory, /<bfm_dir>/BFM.mat
                 data_dir,  # data directory for training and evaluating, /<data_dir>/train/, /<data_dir>/test/, /<data_dir>/meta.json
                 model_dir,  # model directory for saving trained model
                 epochs=1,  # number of epochs for training
                 train_batch_size=16,  # batch size for training
                 eval_batch_size=16,  # batch size for evaluating
                 steps_per_loop=1,  # steps per loop, for efficiency
                 initial_lr=0.001,  # initial learning rate
                 init_checkpoint=None,  # initial checkpoint to restore model if provided
                 init_model_weight_path=None,
                 # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
                 resolution=224,  # image resolution
                 num_gpu=1,  # number of gpus
                 stage='SUPERVISED',  # stage name
                 backbone='resnet50',  # model architecture
                 distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
                 run_eagerly=True,
                 n_tex_para=40, # number of texture parameters used in BFM model
                 model_output_size=426, # model output size, total number of face parameters
                 drange_net=[-1, 1] # dynamic range for input images
                 ):
        # load meta data
        with open(os.path.join(data_dir, 'meta.json'), 'r') as f:
            input_meta_data = json.load(f)
            self.train_data_size = input_meta_data['train_data_size']
            self.eval_data_size = input_meta_data['test_data_size']
            logging.info('Training dataset size: %d' % self.train_data_size)
            logging.info('Evaluating dataset size: %d' % self.eval_data_size)

        self.output_size = model_output_size
        logging.info('Face model parameter size: %d' % self.output_size)

        self.stage = stage
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.steps_per_loop = steps_per_loop

        self.steps_per_epoch = int(self.train_data_size / self.train_batch_size)
        self.total_training_steps = self.steps_per_epoch * self.epochs
        self.eval_steps = int(math.ceil(self.eval_data_size / self.eval_batch_size))

        logging.info('%s Epochs: %d' % (self.stage, self.epochs))
        logging.info('%s Training batch size: %d' % (self.stage, self.train_batch_size))
        logging.info('%s Evaluating batch size: %d' % (self.stage, self.eval_batch_size))
        logging.info('%s Steps per loop: %d' % (self.stage, self.steps_per_loop))
        logging.info('%s Steps per epoch: %d' % (self.stage, self.steps_per_epoch))
        logging.info('%s Total training steps: %d' % (self.stage, self.total_training_steps))

        self.init_checkpoint = init_checkpoint
        self.init_model_weight_path = init_model_weight_path
        self.data_dir = data_dir
        self.train_dir = None
        self.eval_dir = None

        if self.init_checkpoint:
            logging.info('Initial checkpoint: %s' % self.init_checkpoint)
        if self.init_model_weight_path:
            logging.info('Initial model weight path: %s' % self.init_model_weight_path)

        self.resolution = resolution
        self.drange_net = drange_net
        self.num_gpu = num_gpu

        self.backbone = backbone
        self.distribute_strategy = distribute_strategy

        logging.info('Image resolution: %d' % self.resolution)
        logging.info('Number of gpus: %d' % self.num_gpu)
        logging.info('Model backbone: %s' % self.backbone)
        logging.info('Distributed strategy: %s' % self.distribute_strategy)

        if self.distribute_strategy.lower() == 'mirror':
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

        self.train_dataset = None  # training dataset
        self.eval_dataset = None  # testing dataset

        self.model_dir = model_dir  # directory to store trained model and summary
        self.model = None
        self.optimizer = None
        self.initial_lr = initial_lr
        self.train_loss_metric = None
        self.eval_loss_metric = None
        self.use_float16 = False  # whether to use float16
        self.summary_dir = None  # tensorflow summary folder
        self.eval_summary_writer = None  # tensorflow summary writer

        self.run_eagerly = run_eagerly  # whether to run training eagerly

        self.bfm_dir = bfm_dir
        self.n_tex_para = n_tex_para
        self.bfm = None

        logging.info('Initial learning rate: %f' % self.initial_lr)
        logging.info('Run eagerly: %s' % self.run_eagerly)

        self.setup_model_dir()

    def setup_model_dir(self):
        # check if model directory exits, if so, raise error
        if os.path.isdir(self.model_dir):
            if os.listdir(self.model_dir):
                raise ValueError('`model_dir` already exits: %s' % self.model_dir)
        else:
            os.makedirs(self.model_dir)

    def init_bfm(self):
        bfm_path = os.path.join(self.bfm_dir, 'BFM.mat')
        self.bfm = TfMorphableModel(
            model_path=bfm_path,
            n_tex_para=self.n_tex_para
        )

    def create_dataset(self):
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.eval_dir = os.path.join(self.data_dir, 'test')
        self.train_dataset = TFRecordDataset(
            tfrecord_dir=self.train_dir,
            resolution=self.resolution,
            max_label_size='full',
            repeat=True,
            batch_size=self.train_batch_size,
            num_gpu=self.num_gpu
        )
        self.train_dataset.reset_iterator(self.strategy)
        self.eval_dataset = TFRecordDataset(
            tfrecord_dir=self.eval_dir,
            resolution=self.resolution,
            max_label_size='full',
            repeat=False,
            batch_size=self.eval_batch_size,
            num_gpu=self.num_gpu
        )
        self.eval_dataset.reset_iterator(self.strategy)

    def _get_model(self):
        if self.backbone == 'resnet50':
            return Resnet50(image_size=self.resolution, num_output=self.output_size)
        elif self.backbone == 'resnet18':
            return Resnet18(image_size=self.resolution, num_output=self.output_size)
        else:
            raise ValueError('`backbone` not supported: %s' % self.backbone)

    def init_optimizer(self):
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=self.total_training_steps,
            end_learning_rate=0.0)
        self.optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=['layer_norm', 'bias'])

    def init_model(self):

        with distribution_utils.get_strategy_scope(self.strategy):
            # To correctly place the model weights on accelerators,
            # model and optimizer should be created in scope.
            self.model = self._get_model()
            self.init_optimizer()
            self.model.optimizer = self.optimizer

            if self.init_checkpoint:
                logging.info(
                    'Checkpoint file %s found and restoring.', self.init_checkpoint)
                checkpoint = tf.train.Checkpoint(model=self.model)
                checkpoint.restore(self.init_checkpoint).assert_existing_objects_matched()
                logging.info('Loading from checkpoint file completed')
            elif self.init_model_weight_path:
                logging.info(
                    'Model weights file %s found and restoring', self.init_model_weight_path)
                self.model.load_weights(self.init_model_weight_path, by_name=True)
                logging.info('Loading from model weights file completed')

    def train(self):
        logging.info('%s Setting up model directory ...' % self.stage)
        self.setup_model_dir()

        logging.info('%s Initializing logs ...' % self.stage)
        self.init_logs()

        logging.info('%s Creating data ...' % self.stage)
        self.create_dataset()

        logging.info('%s Initializing model ...' % self.stage)
        self.init_model()

        logging.info('%s Setting up metrics ...' % self.stage)
        self.init_metrics()

        logging.info('%s Initializing face model ...' % self.stage)
        self.init_bfm()

        logging.info('%s Starting customized training ...' % self.stage)
        self.run_customized_training_steps()

    def get_loss(self, gt_images, gt_params, est_params, batch_size):
        # split params and unnormalize params
        _, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex = split_300W_LP_labels(gt_params)

        _, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex = unnormalize_labels(
            self.bfm, batch_size, self.resolution, None, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex)

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_300W_LP_labels(est_params)

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = unnormalize_labels(
            self.bfm, batch_size, self.resolution, None, est_lm, est_pp, est_shape, est_exp, est_color, est_illum,
            est_tex)

        # geo loss, render with estimated geo parameters and ground truth pose
        est_geo_gt_pose_images = render_batch(
            pose_param=gt_pp,
            shape_param=est_shape,
            exp_param=est_exp,
            tex_param=est_tex,
            color_param=est_color,
            illum_param=est_illum,
            frame_width=self.resolution,
            frame_height=self.resolution,
            tf_bfm=self.bfm,
            batch_size=batch_size
        )
        loss_pose = tf.reduce_mean(tf.square(gt_images - est_geo_gt_pose_images))

        # pose loss, render with estimated pose parameters and ground truth geo parameters
        gt_geo_est_pose_images = render_batch(
            pose_param=est_pp,
            shape_param=gt_shape,
            exp_param=gt_exp,
            tex_param=gt_tex,
            color_param=gt_color,
            illum_param=gt_illum,
            frame_width=self.resolution,
            frame_height=self.resolution,
            tf_bfm=self.bfm,
            batch_size=batch_size
        )
        loss_geo = tf.reduce_mean(tf.square(gt_images - gt_geo_est_pose_images))

        coef = tf.constant(loss_geo / (loss_pose + loss_geo))
        return (coef * loss_pose + (1 - coef) * loss_geo) / self.strategy.num_replicas_in_sync

    def init_metrics(self):
        self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

    def init_logs(self):
        self.summary_dir = os.path.join(self.model_dir, 'summaries')
        self.eval_summary_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, 'eval'))

    def _replicated_step(self, inputs):
        reals, labels = inputs
        reals_input = process_reals(x=reals, mirror_augment=False, drange_data=self.train_dataset.dynamic_range, drange_net=self.drange_net)
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals_input, training=True)
            loss = self.get_loss(reals, labels, model_outputs, self.train_batch_size)
            if self.use_float16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_float16:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metric.update_state(loss)

    @tf.function
    def _train_steps(self, iterator, steps):
        if not isinstance(steps, tf.Tensor):
            raise ValueError('`steps` should be an Tensor. Python objects can cause retracing')

        for _ in tf.range(steps):
            self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    def _train_single_step(self, iterator):
        self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    @tf.function
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, labels = inputs
            reals_input = process_reals(x=reals, mirror_augment=False, drange_data=self.eval_dataset.dynamic_range, drange_net=self.drange_net)
            model_outputs = self.model(reals_input, training=False)
            loss = self.get_loss(reals, labels, model_outputs, self.eval_batch_size)
            self.eval_loss_metric.update_state(loss)

        self.strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))

    def _run_evaluation(self, current_training_step, test_iterator):
        while True:
            try:
                self._test_step(test_iterator)
            except tf.errors.OutOfRangeError:
                break

        with self.eval_summary_writer.as_default():
            eval_loss = float_metric_value(self.eval_loss_metric)
            logging.info('%s Step: [%d] Validation %s = %f' % (self.stage, current_training_step,
                         self.eval_loss_metric.name, eval_loss))
            self.eval_summary_writer.flush()

    def run_customized_training_steps(self):
        assert tf.executing_eagerly()

        self.use_float16 = isinstance(self.optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)

        if not self.run_eagerly:
            train_single_step = tf.function(self._train_single_step)
        else:
            train_single_step = self._train_single_step

        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(self.model_dir)
        if latest_checkpoint_file:
            logging.info('%s Checkpoint file %s found and restoring from checkpoint' % (self.stage, latest_checkpoint_file))
            checkpoint.restore(latest_checkpoint_file)
            logging.info('%s Loading from checkpoint file completed' % self.stage)
        current_step = self.optimizer.iterations.numpy()
        checkpoint_name = 'rtl_step_{step}.ckpt'

        while current_step < self.total_training_steps:
            self.train_loss_metric.reset_states()

            steps = steps_to_run(current_step, self.steps_per_epoch, self.steps_per_loop)

            if steps == 1:
                train_single_step(self.train_dataset)
            else:
                self._train_steps(self.train_dataset, tf.convert_to_tensor(steps, dtype=tf.int32))

            current_step += steps

            train_loss = float_metric_value(self.train_loss_metric)
            training_status = '%s Train Step: %d/%d  / loss = %s' % (self.stage, current_step, self.total_training_steps, train_loss)

            logging.info(training_status)

            if current_step % self.steps_per_epoch == 0:
                if current_step < self.total_training_steps:
                    save_checkpoint(checkpoint, self.model_dir, checkpoint_name.format(step=current_step))

                if self.eval_dataset:
                    logging.info('%s Running evaluation after step: %s' % (self.stage, current_step))
                    self.eval_dataset.reset_iterator(self.strategy)
                    self._run_evaluation(current_step, self.eval_dataset)
                    self.eval_loss_metric.reset_states()

        save_checkpoint(checkpoint, self.model_dir, checkpoint_name.format(step=current_step))

        if self.eval_dataset:
            logging.info('%s Running final evaluation after training is complete.' % self.stage)
            self.eval_dataset.reset_iterator(self.strategy)
            self._run_evaluation(current_step, self.eval_dataset)

        training_summary = {
            'stage': self.stage,
            'backbone': self.backbone,
            'init_checkpoint': self.init_checkpoint,
            'init_model_weight_path': self.init_model_weight_path,
            'total_training_steps': self.total_training_steps,
            'train_loss': float_metric_value(self.train_loss_metric)
        }

        if self.eval_dataset:
            training_summary['eval_metric'] = float_metric_value(self.eval_loss_metric)

        write_txt_summary(training_summary, self.summary_dir)


if __name__ == '__main__':

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

    train_model = TrainFaceModelUnsupervised(
        bfm_dir='/opt/data/BFM/',
        data_dir='/opt/data/face-fuse/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/20200310/unsupervised/',  # model directory for saving trained model
        epochs=3,  # number of epochs for training
        train_batch_size=16,  # batch size for training
        eval_batch_size=16,  # batch size for evaluating
        steps_per_loop=1,  # steps per loop, for efficiency
        initial_lr=0.0001,  # initial learning rate
        init_checkpoint=None,  # initial checkpoint to restore model if provided
        init_model_weight_path=None,
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='UNSUPERVISED',  # stage name
        backbone='resnet18',  # model architecture
        distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
        run_eagerly=True,
        model_output_size=426 # number of face parameters, we remove region of interests, roi from the data
    )

    train_model.train()