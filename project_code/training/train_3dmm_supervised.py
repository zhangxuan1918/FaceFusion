import logging

import tensorflow as tf
from absl import flags

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

        logging.info('%s Starting customized training ...' % self.stage)
        self.run_customized_training_steps()

    def get_loss(self, gt, est):
        loss = tf.reduce_mean(tf.square(gt - est))
        return loss / self.strategy.num_replicas_in_sync

    def _replicated_step(self, inputs):
        reals, labels = inputs
        reals = process_reals(x=reals, mirror_augment=False, drange_data=self.train_dataset.dynamic_range,
                              drange_net=self.drange_net)
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals, training=True)
            loss = self.get_loss(labels[:, 4:], model_outputs)
            if self.use_float16:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
        if self.use_float16:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss_metric.update_state(loss)

    def _train_single_step(self, iterator):
        self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    @tf.function
    def _test_step(self, iterator):

        def _test_step_fn(inputs):
            reals, labels = inputs
            reals = process_reals(x=reals, mirror_augment=False, drange_data=self.eval_dataset.dynamic_range,
                                  drange_net=self.drange_net)
            model_outputs = self.model(reals, training=False)
            loss = self.get_loss(labels[:, 4:], model_outputs)
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

    train_model = TrainFaceModelSupervised(
        data_dir='/opt/data/face-fuse/',  # data directory for training and evaluating
        model_dir='/opt/data/face-fuse/model/20200310/supervised/',  # model directory for saving trained model
        epochs=3,  # number of epochs for training
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
        distribute_strategy='mirror',  # distribution strategy when num_gpu > 1
        run_eagerly=True,
        model_output_size=426
    )

    train_model.train()
