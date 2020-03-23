import datetime
import logging

import tensorflow as tf

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, unnormalize_labels
from project_code.misc.image_utils import process_reals
from project_code.training.optimization import AdamWeightDecay
from project_code.training.train_3dmm import TrainFaceModel

logging.basicConfig(level=logging.INFO)


class TrainFaceModelSupervised(TrainFaceModel):

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
            # training loss metric for texture
            'loss_geo': tf.keras.metrics.Mean('train_loss_geo', dtype=tf.float32),
            # training loss metric for landmarks
            'loss_lm': tf.keras.metrics.Mean('train_loss_lm', dtype=tf.float32),
            # training loss metric for shape, exp and pose parameters
            'loss_shape': tf.keras.metrics.Mean('train_loss_shape', dtype=tf.float32)
        }

        # evaluating loss metrics
        self.eval_loss_metrics = {
            # evaluating loss metric for texture
            'loss_geo': tf.keras.metrics.Mean('eval_loss_geo', dtype=tf.float32),
            # evaluating loss metric for landmarks
            'loss_lm': tf.keras.metrics.Mean('eval_loss_lm', dtype=tf.float32),
            # evaluating loss metric for shape, exp and pose parameters
            'loss_shape': tf.keras.metrics.Mean('eval_loss_shape', dtype=tf.float32)
        }

    def get_loss(self, gt_params, est_params, batch_size):
        # gt contains roi, landmarks and all face parameters
        # est only contains face parameters
        # split params and unnormalize params
        _, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex = split_300W_LP_labels(gt_params)
        # unnormalize grountruth landmarks
        gt_lm = tf.reshape(gt_lm, (batch_size, 2, -1)) * self.resolution

        fake_roi, fake_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_300W_LP_labels(est_params)

        # geo/texture related loss
        loss_geo = tf.sqrt(tf.reduce_mean(tf.square(gt_color - est_color)))
        loss_geo += tf.sqrt(tf.reduce_mean(tf.square(gt_illum - est_illum)))
        loss_geo += tf.sqrt(tf.reduce_mean(tf.square(gt_tex - est_tex)))

        loss_shape = tf.sqrt(tf.reduce_mean(tf.square(gt_shape - est_shape)))
        loss_shape += tf.sqrt(tf.reduce_mean(tf.square(gt_exp - est_exp)))
        loss_shape += tf.sqrt(tf.reduce_mean(tf.square(gt_pp - est_pp)))

        # shape related loss, we compute the difference between landmarks
        _, _, est_pp, est_shape, est_exp, _, _, _ = unnormalize_labels(
            self.bfm, batch_size, self.resolution, fake_roi, fake_lm, est_pp, est_shape, est_exp, est_color, est_illum,
            est_tex)

        est_lm = self.bfm.get_landmarks(
            shape_param=est_shape,
            exp_param=est_exp,
            pose_param=est_pp,
            batch_size=batch_size,
            resolution=self.resolution,
            is_2d=True,
            is_plot=True
        )

        loss_lm = tf.sqrt(tf.reduce_mean(tf.square(gt_lm - est_lm))) / self.resolution
        # TODO: try https://www.tensorflow.org/api_docs/python/tf/Variable
        # self.coef_geo = tf.Variable((loss_lm + loss_shape) / (2.0 * loss_total), trainable=False)
        # self.coef_lm = tf.Variable((loss_geo + loss_shape) / (2.0 * loss_total), trainable=False)
        # return (self.coef_geo * loss_geo + self.coef_lm * loss_lm + (1.0 - self.coef_geo - self.coef_lm) * loss_geo) / self.strategy.num_replicas_in_sync
        return loss_geo / self.strategy.num_replicas_in_sync, loss_lm / self.strategy.num_replicas_in_sync, loss_shape / self.strategy.num_replicas_in_sync

    def _replicated_step(self, inputs):
        reals, labels = inputs
        _, reals = process_reals(x=reals, mirror_augment=False, drange_data=self.train_dataset.dynamic_range,
                              drange_net=self.drange_net)
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals, training=True)
            loss_geo, loss_lm, loss_shape = self.get_loss(gt_params=labels, est_params=model_outputs, batch_size=self.train_batch_size)
            loss = loss_geo + loss_lm + loss_shape
            if self.use_float16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_float16:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metrics['loss_geo'].update_state(loss_geo)
        self.train_loss_metrics['loss_lm'].update_state(loss_lm)
        self.train_loss_metrics['loss_shape'].update_state(loss_shape)

    def _train_single_step(self, iterator):
        self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    @tf.function
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, labels = inputs
            reals = process_reals(x=reals, mirror_augment=False, drange_data=self.eval_dataset.dynamic_range,
                                  drange_net=self.drange_net)
            model_outputs = self.model(reals, training=False)

            loss_geo, loss_lm, loss_shape = self.get_loss(gt_params=labels, est_params=model_outputs, batch_size=self.eval_batch_size)
            self.eval_loss_metrics['loss_geo'].update_state(loss_geo)
            self.eval_loss_metrics['loss_lm'].update_state(loss_lm)
            self.eval_loss_metrics['loss_shape'].update_state(loss_shape)

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
        n_tex_para=40,  # number of texture params used
        data_dir='/opt/data/face-fuse/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/{0}/supervised/'.format(date_yyyymmdd),  # model directory for saving trained model
        epochs=10,  # number of epochs for training
        train_batch_size=64,  # batch size for training
        eval_batch_size=64,  # batch size for evaluating
        steps_per_loop=10,  # steps per loop, for efficiency
        initial_lr=0.00005,  # initial learning rate
        init_checkpoint=None,  # initial checkpoint to restore model if provided
        init_model_weight_path='/opt/data/face-fuse/model/face_vgg_v2/weights.h5',
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='UNSUPERVISED',  # stage name
        backbone='resnet50',  # model architecture
        distribute_strategy='one_device',  # distribution strategy when num_gpu > 1
        run_eagerly=False,
        model_output_size=290,
        enable_profiler=False
    )

    train_model.train()
