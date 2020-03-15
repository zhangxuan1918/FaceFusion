import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
import numpy as np


class Toy:

    def __init__(self):
        self.data = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        self.data2 = tf.constant([0.0, 0.2], dtype=tf.float32)

    def multiply(self, input):
        res = tf.expand_dims(self.data2, 0) + tf.einsum('ij,kj->ki', self.data, input)
        return res


class TrainFakeModel():

    def __init__(self):

        self.epochs = 1
        self.steps_per_loop = 10

        self.steps_per_epoch = 1000
        self.total_training_steps = 5000
        self.strategy = tf.distribute.MirroredStrategy()

        self.train_dataset = None  # training dataset
        self.model = None
        self.optimizer = None
        self.initial_lr = 0.00001
        self.toy = Toy()

    def create_dataset(self):
        example_data = np.random.random((128, 10, 10, 3)).astype(np.float32)
        labels = np.random.random((128, 2)).astype(np.float32)
        train_raw = tf.data.Dataset.from_tensor_slices(example_data)
        train_label = tf.data.Dataset.from_tensor_slices(labels)
        train_ds = tf.data.Dataset.zip((train_raw, train_label))
        train_ds = train_ds.batch(2).repeat(10000)
        self.train_dataset = iter(self.strategy.experimental_distribute_dataset(train_ds))

    def _get_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(3)
            ]
        )
        model.build(input_shape=(None, 10, 10, 3))
        model.summary()

        return model

    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.RMSprop(lr=self.initial_lr)

    def init_model(self):

        with self.strategy.scope():
            # To correctly place the model weights on accelerators,
            # model and optimizer should be created in scope.
            self.model = self._get_model()
            self.init_optimizer()
            self.model.optimizer = self.optimizer

    def train(self):
        self.init_model()
        self.create_dataset()
        self.run_customized_training_steps()

    def get_loss(self, gt, est):
        est = self.toy.multiply(est)
        return tf.reduce_mean(tf.square(gt - est))

    def _replicated_step(self, inputs):
        reals, labels = inputs
        with tf.GradientTape() as tape:
            model_outputs = self.model(reals, training=True)
            loss = self.get_loss(labels, model_outputs)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def _train_steps(self, iterator, steps):

        for _ in tf.range(steps):
            self.strategy.experimental_run_v2(self._replicated_step, args=(next(iterator),))

    def run_customized_training_steps(self):
        current_step = self.optimizer.iterations.numpy()

        while current_step < self.total_training_steps:
            self._train_steps(self.train_dataset, tf.convert_to_tensor(self.steps_per_epoch, dtype=tf.int32))

            current_step += self.steps_per_epoch

            print(current_step)


if __name__ == '__main__':
    trainmodel = TrainFakeModel()
    trainmodel.train()
