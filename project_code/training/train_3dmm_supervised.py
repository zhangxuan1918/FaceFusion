import json
import logging
import math
import os

import tensorflow as tf
from absl import flags

from project_code.misc import distribution_utils
from project_code.misc.train_utils import float_metric_value, steps_to_run, save_checkpoint, write_txt_summary
from project_code.models.resnet18 import Resnet18
from project_code.models.resnet50 import Resnet50
from project_code.training.dataset import TFRecordDataset
from project_code.training.optimization import AdamWeightDecay

flags.DEFINE_string('train_data_path', None, 'Path to training data for 3DMM')
flags.DEFINE_string('eval_data_path', None, 'Path to evaluating data for 3DMM')

flags.DEFINE_string(
    'input_meta_data_path', None, 'Path to file that contains meta dta about training and evaluating data'
)


class TrainFaceModelSupervised:

    def __init__(self,
                 input_meta_data_path, # input meta data json path, contains number of training data, number of evaluating data
                 data_dir,  # data directory for training and evaluating
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
                 run_eagerly=True
                 ):
        with open(input_meta_data_path, 'r') as f:
            input_meta_data = json.load(f)

        self.output_size = input_meta_data['num_output']  # output size, number of face params
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.steps_per_loop = steps_per_loop

        self.steps_per_epoch = int(input_meta_data['train_data_size'] / self.train_batch_size)
        self.total_training_steps = self.steps_per_epoch * self.epochs
        self.eval_steps = int(math.ceil(input_meta_data['eval_data_size'] / self.eval_batch_size))

        self.initial_lr = initial_lr
        self.init_checkpoint = init_checkpoint
        self.init_model_weight_path = init_model_weight_path
        self.data_dir = data_dir
        self.train_dir = None
        self.eval_dir = None

        self.resolution = resolution
        self.num_gpu = num_gpu
        self.stage = stage
        self.backbone = backbone
        self.distribute_strategy = distribute_strategy

        if self.distribute_strategy.lower() == 'mirror':
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

        self.train_dataset = None  # training dataset
        self.eval_dataset = None  # testing dataset

        self.model_dir = model_dir  # directory to store trained model and summary
        self.model = None
        self.optimizer = None
        self.train_loss_metric = None
        self.eval_loss_metric = None
        self.use_float16 = False  # whether to use float16
        self.summary_dir = None  # tensorflow summary folder
        self.eval_summary_writer = None  # tensorflow summary writer

        self.run_eagerly = run_eagerly  # whether to run training eagerly

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
                # TODO load weights
                pass

    def train(self):
        logging.info('Creating data ...')
        self.create_dataset()

        logging.info('Initializing model ...')
        self.init_model()

        logging.info('Setting up metrics ...')
        self.init_metrics()

        logging.info('Starting customized training ...')
        self.run_customized_training_steps()

    def get_loss(self, gt, est):
        loss = tf.reduce_mean(tf.square(gt - est))
        return loss / self.strategy.num_replicas_in_sync

    def init_metrics(self):
        self.train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.eval_loss_metric = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)

    def init_logs(self):
        self.summary_dir = os.path.join(self.model_dir, 'summaries')
        self.eval_summary_writer = tf.summary.create_file_writer(os.path.join(self.summary_dir, 'eval'))

    def _replicated_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            model_outputs = self.model(inputs, training=True)
            loss = self.get_loss(labels, model_outputs)
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
            inputs, labels = inputs
            model_outputs = self.model(inputs, training=False)
            loss = self.get_loss(labels, model_outputs)
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
            logging.info('Step: [%d] Validation %s = %f', current_training_step,
                         self.eval_loss_metric.name, eval_loss)
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
            logging.info('Checkpoint file %s found and restoring from checkpoint', latest_checkpoint_file)
            checkpoint.restore(latest_checkpoint_file)
            logging.info('Loading from checkpoint file completed')
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
            training_status = 'Train Step: %d/%d  / loss = %s' % (current_step, self.total_training_steps, train_loss)

            logging.info(training_status)

            if current_step % self.steps_per_epoch == 0:
                if current_step < self.total_training_steps:
                    save_checkpoint(checkpoint, self.model_dir, checkpoint_name.format(step=current_step))

                if self.eval_dataset:
                    logging.info('Running evaluation after step: %s', current_step)
                    self.eval_dataset.reset_iterator()
                    self._run_evaluation(current_step, self.eval_dataset)
                    self.eval_loss_metric.reset_states()

        save_checkpoint(checkpoint, self.model_dir, checkpoint_name.format(step=current_step))

        if self.eval_dataset:
            logging.info('Running final evaluation after training is complete.')
            self.eval_dataset.reset_iterator()
            self._run_evaluation(current_step, self.eval_dataset)

        training_summary = {
            'total_training_steps': self.total_training_steps,
            'train_loss': float_metric_value(self.train_loss_metric)
        }

        if self.eval_dataset:
            training_summary['eval_metric'] = float_metric_value(self.eval_loss_metric)

        write_txt_summary(training_summary, self.summary_dir)
