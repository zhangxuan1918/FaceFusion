import datetime
import logging
import os

import tensorflow as tf
from tf_3dmm.mesh.render import render_batch

from project_code.create_tfrecord.export_tfrecord_util import split_ffhq_labels
from project_code.misc.image_utils import process_reals_supervised
from project_code.training.dataset import TFRecordDatasetSupervised
from project_code.training.optimization import AdamWeightDecay
from project_code.training.train_3dmm import TrainFaceModel

logging.basicConfig(level=logging.INFO)


class TrainFaceModelSupervised(TrainFaceModel):

    def create_training_dataset(self):
        if self.train_dir is None:
            self.train_dir = os.path.join(self.data_dir, 'train')
        if self.eval_dir is None:
            self.eval_dir = os.path.join(self.data_dir, 'test')

        self.train_dataset = TFRecordDatasetSupervised(
            tfrecord_dir=self.train_dir,
            resolution=self.resolution,
            repeat=True,
            batch_size=self.train_batch_size,
            num_gpu=self.num_gpu,
            strategy=self.strategy,
            is_augment=self.is_augment
        )

    def create_evaluating_dataset(self):
        if self.eval_dir is None:
            self.eval_dir = os.path.join(self.data_dir, 'test')
        self.eval_dataset = TFRecordDatasetSupervised(
            tfrecord_dir=self.eval_dir,
            resolution=self.resolution,
            repeat=False,
            batch_size=self.eval_batch_size,
            num_gpu=self.num_gpu,
            strategy=self.strategy,
            is_augment=False
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
        # workaround for loss computation
        # self.coef_geo = None
        # self.coef_lm = None
        self.run_customized_training_steps()

    def init_metrics(self):
        # training loss metrics
        self.train_loss_metrics = {
            # training loss metric for images
            'loss_img': tf.keras.metrics.Mean('train_loss_img', dtype=tf.float32),
            # training loss metric for landmarks
            'loss_lms': tf.keras.metrics.Mean('train_loss_lms', dtype=tf.float32),
            # training loss metric for landmarks
            'loss_reg': tf.keras.metrics.Mean('train_loss_reg', dtype=tf.float32),
        }

        # evaluating loss metrics
        self.eval_loss_metrics = {
            # evaluating loss metric for images
            'loss_img': tf.keras.metrics.Mean('eval_loss_img', dtype=tf.float32),
            # evaluating loss metric for landmarks
            'loss_lms': tf.keras.metrics.Mean('eval_loss_lms', dtype=tf.float32),
        }

    def get_loss(self, gt_params, gt_images, est_params, batch_size):
        # gt_params have only landmarks
        gt_lm = tf.reshape(gt_params, shape=(-1, 2, 68)) * self.resolution

        est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_ffhq_labels(est_params)

        # regularization loss
        loss_reg = tf.sqrt(tf.reduce_mean(tf.square(est_pp)))
        loss_reg += tf.sqrt(tf.reduce_mean(tf.square(est_shape)))
        loss_reg += tf.sqrt(tf.reduce_mean(tf.square(est_exp)))
        loss_reg += tf.sqrt(tf.reduce_mean(tf.square(est_color)))
        loss_reg += tf.sqrt(tf.reduce_mean(tf.square(est_illum)))
        loss_reg += tf.sqrt(tf.reduce_mean(tf.square(est_tex)))

        est_pp, est_shape, est_exp, est_color, est_illum, est_tex = self.unnormalize_labels(
            batch_size, est_pp, est_shape, est_exp, est_color, est_illum, est_tex)

        # add 0 to t3d z axis
        # only have x, y translation
        est_pp = tf.concat([est_pp[:, :-1], tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float32), est_pp[:, -1:]], axis=1)

        # image rendered with ground truth shape param, loss on texture/color
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
        gt_images = tf.cast(tf.where(est_images > 0, gt_images, 0), tf.float32)
        loss_img = tf.sqrt(tf.reduce_mean(tf.square(est_images - gt_images)))

        # landmark loss
        est_lm = self.bfm.get_landmarks(
            shape_param=est_shape,
            exp_param=est_exp,
            pose_param=est_pp,
            batch_size=batch_size,
            resolution=self.resolution,
            is_2d=True,
            is_plot=True
        )

        loss_lms = tf.sqrt(tf.reduce_mean(tf.square(gt_lm - est_lm)))

        return loss_img / self.strategy.num_replicas_in_sync, 50.0 * loss_lms / self.strategy.num_replicas_in_sync, loss_reg / self.strategy.num_replicas_in_sync

    def _replicated_step(self, inputs):
        reals, labels = inputs
        reals_input = process_reals_supervised(x=reals, mirror_augment=False,
                                               drange_data=self.train_dataset.dynamic_range,
                                               drange_net=self.drange_net)
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals_input, training=True)

            loss_img, loss_lms, loss_reg = self.get_loss(gt_params=labels, gt_images=reals, est_params=model_outputs,
                                                         batch_size=self.train_batch_size)
            loss = loss_img + loss_lms + loss_reg
            if self.use_float16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_float16:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metrics['loss_img'].update_state(loss_img)
        self.train_loss_metrics['loss_lms'].update_state(loss_lms)
        self.train_loss_metrics['loss_reg'].update_state(loss_reg)

    def _train_single_step(self, iterator):
        self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    @tf.function
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, labels = inputs
            reals_input = process_reals_supervised(x=reals, mirror_augment=False,
                                                   drange_data=self.eval_dataset.dynamic_range,
                                                   drange_net=self.drange_net)
            model_outputs = self.model(reals_input, training=False)

            loss_img, loss_lms, _ = self.get_loss(gt_params=labels, gt_images=reals, est_params=model_outputs,
                                                  batch_size=self.train_batch_size)

            self.eval_loss_metrics['loss_img'].update_state(loss_img)
            self.eval_loss_metrics['loss_lms'].update_state(loss_lms)

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

    date_yyyymmdd = datetime.datetime.today().strftime('%Y%m%d')
    train_model = TrainFaceModelSupervised(
        bfm_dir='/opt/data/BFM/',
        exp_path='/opt/data/face-fuse/exp_80k.npz',
        param_mean_std_path='/opt/data/face-fuse/stats_80k.npz',
        n_tex_para=40,  # number of texture params used
        n_shape_para=100, # number of shape params used
        data_dir='/opt/data/face-fuse/supervised_ffhq_arg/',  # data directory for training and evaluating
        is_augment=True,
        model_dir='/opt/data/face-fuse/model/{0}/supervised/'.format(date_yyyymmdd),
        # model directory for saving trained model
        epochs=25,  # number of epochs for training
        train_batch_size=32,  # batch size for training
        eval_batch_size=32,  # batch size for evaluating
        initial_lr=0.00005,  # initial learning rate
        init_checkpoint=None,  # initial checkpoint to restore model if provided
        init_model_weight_path=None,  # '/opt/data/face-fuse/model/face_vgg_v2/weights.h5',
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='SUPERVISED',  # stage name
        backbone='resnet50',  # model architecture
        distribute_strategy='one_device',  # distribution strategy when num_gpu > 1
        run_eagerly=False,
        steps_per_loop=100,  # steps per loop, for efficiency
        model_output_size=240,
        enable_profiler=False,
        data_name='FFHQ'# which dataset to use, 300W_LP or 80K or FFHQ,
    )

    train_model.train()
