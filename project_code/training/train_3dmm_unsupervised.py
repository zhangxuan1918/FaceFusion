import logging
import os

import tensorflow as tf
from tf_3dmm.mesh.reader import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, \
    unnormalize_labels
from project_code.misc.image_utils import process_reals
from project_code.training.optimization import AdamWeightDecay
from project_code.training.train_3dmm import TrainFaceModel

logging.basicConfig(level=logging.INFO)


class TrainFaceModelUnsupervised(TrainFaceModel):

    def __init__(self, bfm_dir, data_dir, model_dir, epochs=1, train_batch_size=16, eval_batch_size=16,
                 steps_per_loop=1, initial_lr=0.001, init_checkpoint=None, init_model_weight_path=None, resolution=224,
                 num_gpu=1, stage='SUPERVISED', backbone='resnet50', distribute_strategy='mirror', run_eagerly=True,
                 n_tex_para=40, model_output_size=426, drange_net=[-1, 1]):
        # load meta data

        super().__init__(data_dir, model_dir, epochs, train_batch_size, eval_batch_size, steps_per_loop, initial_lr,
                         init_checkpoint, init_model_weight_path, resolution, num_gpu, stage, backbone,
                         distribute_strategy, run_eagerly, model_output_size, drange_net)

        self.bfm_dir = bfm_dir
        self.n_tex_para = n_tex_para
        self.bfm = None

    def init_bfm(self):
        bfm_path = os.path.join(self.bfm_dir, 'BFM.mat')
        self.bfm = TfMorphableModel(
            model_path=bfm_path,
            n_tex_para=self.n_tex_para
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

        coef = (loss_geo / (loss_pose + loss_geo))
        return (coef * loss_pose + (1 - coef) * loss_geo) / self.strategy.num_replicas_in_sync

    def _replicated_step(self, inputs):
        reals, labels = inputs
        reals_input = process_reals(x=reals, mirror_augment=False, drange_data=self.train_dataset.dynamic_range,
                                    drange_net=self.drange_net)
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals_input, training=True)
            loss = self.get_loss(tf.cast(reals, tf.float32), labels, model_outputs, self.train_batch_size)
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
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, labels = inputs
            reals_input = process_reals(x=reals, mirror_augment=False, drange_data=self.eval_dataset.dynamic_range,
                                        drange_net=self.drange_net)
            model_outputs = self.model(reals_input, training=False)
            loss = self.get_loss(tf.cast(reals, tf.float32), labels, model_outputs, self.eval_batch_size)
            self.eval_loss_metric.update_state(loss)

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

    train_model = TrainFaceModelUnsupervised(
        bfm_dir='/opt/data/BFM/',
        data_dir='/opt/data/face-fuse/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/20200310/unsupervised/',  # model directory for saving trained model
        epochs=3,  # number of epochs for training
        train_batch_size=16,  # batch size for training
        eval_batch_size=16,  # batch size for evaluating
        steps_per_loop=10,  # steps per loop, for efficiency
        initial_lr=0.0005,  # initial learning rate
        init_checkpoint='/opt/data/face-fuse/model/20200310/supervised/',  # initial checkpoint to restore model if provided
        init_model_weight_path=None,
        # initial model weight to use if provided, if init_checkpoint is provided, this param will be ignored
        resolution=224,  # image resolution
        num_gpu=1,  # number of gpus
        stage='UNSUPERVISED',  # stage name
        backbone='resnet50',  # model architecture
        distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
        run_eagerly=True,
        model_output_size=426  # number of face parameters, we remove region of interests, roi from the data
    )

    train_model.train()
