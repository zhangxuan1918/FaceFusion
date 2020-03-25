import logging
import os

import tensorflow as tf
from tf_3dmm.mesh.render import render_batch

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, unnormalize_labels
from project_code.misc.image_utils import process_reals_unsupervised
from project_code.training.dataset import TFRecordDatasetUnsupervised
from project_code.training.optimization import AdamWeightDecay
from project_code.training.train_3dmm import TrainFaceModel

logging.basicConfig(level=logging.INFO)


class TrainFaceModelUnsupervised(TrainFaceModel):

    def create_training_dataset(self):
        if self.train_dir is None:
            self.train_dir = os.path.join(self.data_dir, 'train')
        if self.eval_dir is None:
            self.eval_dir = os.path.join(self.data_dir, 'test')

        self.train_dataset = TFRecordDatasetUnsupervised(
            tfrecord_dir=self.train_dir,
            resolution=self.resolution,
            repeat=True,
            batch_size=self.train_batch_size,
            num_gpu=self.num_gpu,
            strategy=self.strategy
        )

    def create_evaluating_dataset(self):
        if self.eval_dir is None:
            self.eval_dir = os.path.join(self.data_dir, 'test')
        self.eval_dataset = TFRecordDatasetUnsupervised(
            tfrecord_dir=self.eval_dir,
            resolution=self.resolution,
            repeat=True,
            batch_size=self.eval_batch_size,
            num_gpu=self.num_gpu,
            strategy=self.strategy
        )

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

    def train(self):
        logging.info('%s Setting up model directory ...' % self.stage)
        self.setup_model_dir()

        logging.info('%s Initializing logs ...' % self.stage)
        self.init_logs()

        logging.info('%s Creating training data ...' % self.stage)
        self.create_training_dataset()

        logging.info('%s Initializing model ...' % self.stage)
        self.init_model()

        logging.info('%s Setting up metrics ...' % self.stage)
        self.init_metrics()

        logging.info('%s Initializing face model ...' % self.stage)
        self.init_bfm()

        logging.info('%s Starting customized training ...' % self.stage)

        if self.enable_profiler:
            os.makedirs(os.path.join(self.summary_dir, 'profiler'))
            from tensorflow.python.eager import profiler
            profiler.start_profiler_server(6019)
        self.run_customized_training_steps()

    def init_metrics(self):
        # training loss metrics
        self.train_loss_metrics = {
            # training loss metric
            'loss': tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
        }

        # evaluating loss metrics
        self.eval_loss_metrics = {
            # evaluating loss metric for texture
            'loss': tf.keras.metrics.Mean('eval_loss', dtype=tf.float32),
        }

    def get_loss(self, gt_images, gt_mask, est_params, batch_size):
        # split params and unnormalize params

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_300W_LP_labels(est_params)

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = unnormalize_labels(
            self.bfm, batch_size, self.resolution, None, est_lm, est_pp, est_shape, est_exp, est_color, est_illum,
            est_tex)

        # geo loss, render with estimated geo parameters and ground truth pose
        est_images = render_batch(
            pose_param=est_pp,
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

        gt_images = tf.where(gt_mask == 255, gt_images, 0)
        est_images = tf.where(gt_mask == 255, est_images, 0)

        loss = tf.sqrt(tf.reduce_mean(tf.square(gt_images - est_images)))

        return loss / self.strategy.num_replicas_in_sync
        # if loss_geo too big, make is smaller, so we balance the loss between geo and pose
        # coef = tf.Variable((loss_geo / (loss_pose + loss_geo)))
        # return ((1 - coef) * loss_geo + coef * loss_pose) / self.strategy.num_replicas_in_sync

    def _replicated_step(self, inputs):
        reals, masks = inputs
        reals_input, masks = process_reals_unsupervised(images=reals, masks=masks, mirror_augment=False,
                                                        drange_data=self.train_dataset.dynamic_range,
                                                        drange_net=self.drange_net, batch_size=self.train_batch_size,
                                                        resolution=self.resolution)

        with tf.GradientTape() as tape:
            model_outputs = self.model(reals_input, training=True)
            loss = self.get_loss(reals, masks, model_outputs, self.train_batch_size)
            if self.use_float16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_float16:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metrics['loss'].update_state(loss)

    @tf.function
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, masks = inputs

            reals_input, masks = process_reals_unsupervised(images=reals, masks=masks, mirror_augment=False,
                                                            drange_data=self.train_dataset.dynamic_range,
                                                            drange_net=self.drange_net, batch_size=self.train_batch_size,
                                                            resolution=self.resolution)

            model_outputs = self.model(reals_input, training=False)

            loss = self.get_loss(reals, masks, model_outputs, self.eval_batch_size)
            self.eval_loss_metrics['loss'].update_state(loss)

        self.strategy.experimental_run_v2(_test_step_fn, args=(next(iterator),))


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

    date_yyyymmdd = '20200322'
    train_model = TrainFaceModelUnsupervised(
        bfm_dir='/opt/data/BFM/',
        n_tex_para=40,  # number of texture params used
        data_dir='/opt/data/face-fuse/unsupervised/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/{0}/unsupervised/'.format(date_yyyymmdd),  # model directory for saving trained model
        epochs=10,  # number of epochs for training
        train_batch_size=64,  # batch size for training
        eval_batch_size=64,  # batch size for evaluating
        steps_per_loop=10,  # steps per loop, for efficiency
        initial_lr=0.00005,  # initial learning rate
        init_checkpoint='/opt/data/face-fuse/model/{0}/supervised/'.format(date_yyyymmdd),
        # initial checkpoint to restore model if provided
        init_model_weight_path=None,
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='UNSUPERVISED',  # stage name
        backbone='resnet50',  # model architecture
        # distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
        distribute_strategy='one_device',
        run_eagerly=False,
        model_output_size=290,  # number of face parameters, we remove region of interests, roi from the data
        enable_profiler=False
    )

    train_model.train()
