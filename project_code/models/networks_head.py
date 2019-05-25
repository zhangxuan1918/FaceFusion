from tensorflow.python import keras
import tensorflow as tf


class Head3dmm(keras.Model):

    def __init__(self, output_channel: int, header_name):
        super(Head3dmm, self).__init__()
        self.output_channel = output_channel
        self.conv = keras.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_normal',
            name='conv_{0}'.format(header_name)
        )
        self.avg_pool = keras.layers.AveragePooling2D(
            pool_size=(7, 7),
            name='avg_pool_{0}'.format(header_name)

        )
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(
            units=self.output_channel,
            activation=None,
            use_bias=True,
            name='dense_{0}'.format(header_name)
        )

    def call(self, inputs, training=True):
        # input size (7, 7)
        x = self.conv(inputs)
        x = self.avg_pool(x)
        x = self.dense(x)

        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.output_channel])