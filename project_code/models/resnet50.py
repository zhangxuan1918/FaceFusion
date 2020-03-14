from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense
from tensorflow.python.keras.regularizers import l2

from project_code.models.resnet_uil import conv_block_v2, identity_block_v2, WEIGHT_DECAY


class Resnet50(tf.keras.Model):

    def __init__(self, image_size, num_output, **kwargs):
        self.image_size = image_size
        self.num_output = num_output
        self.__config = {
            'image_size': image_size,
            'num_output': num_output
        }

        inputs = Input(shape=[self.image_size, self.image_size, 3], name='image_input')
        x = self.resnet50_backend(inputs=inputs)

        # AvgPooling
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(self.num_output, name='fc')(x)

        super(Resnet50, self).__init__(
            inputs=inputs, outputs=x, **kwargs
        )

    def resnet50_backend(self, inputs):
        bn_axis = 3
        # inputs are of size 224 x 224 x 3
        x = Conv2D(64, (7, 7), strides=(2, 2),
                   kernel_initializer='orthogonal',
                   use_bias=False,
                   trainable=True,
                   kernel_regularizer=l2(WEIGHT_DECAY),
                   padding='same',
                   name='conv1/7x7_s2')(inputs)

        # inputs are of size 112 x 112 x 64
        x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # inputs are of size 56 x 56
        x = conv_block_v2(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), trainable=True)
        x = identity_block_v2(x, 3, [64, 64, 256], stage=2, block=2, trainable=True)
        x = identity_block_v2(x, 3, [64, 64, 256], stage=2, block=3, trainable=True)

        # inputs are of size 28 x 28
        x = conv_block_v2(x, 3, [128, 128, 512], stage=3, block=1, trainable=True)
        x = identity_block_v2(x, 3, [128, 128, 512], stage=3, block=2, trainable=True)
        x = identity_block_v2(x, 3, [128, 128, 512], stage=3, block=3, trainable=True)
        x = identity_block_v2(x, 3, [128, 128, 512], stage=3, block=4, trainable=True)

        # inputs are of size 14 x 14
        x = conv_block_v2(x, 3, [256, 256, 1024], stage=4, block=1, trainable=True)
        x = identity_block_v2(x, 3, [256, 256, 1024], stage=4, block=2, trainable=True)
        x = identity_block_v2(x, 3, [256, 256, 1024], stage=4, block=3, trainable=True)
        x = identity_block_v2(x, 3, [256, 256, 1024], stage=4, block=4, trainable=True)
        x = identity_block_v2(x, 3, [256, 256, 1024], stage=4, block=5, trainable=True)
        x = identity_block_v2(x, 3, [256, 256, 1024], stage=4, block=6, trainable=True)

        # inputs are of size 7 x 7
        x = conv_block_v2(x, 3, [512, 512, 2048], stage=5, block=1, trainable=True)
        x = identity_block_v2(x, 3, [512, 512, 2048], stage=5, block=2, trainable=True)
        x = identity_block_v2(x, 3, [512, 512, 2048], stage=5, block=3, trainable=True)
        return x


if __name__ == '__main__':
    resnet50 = Resnet50(image_size=224, num_output=450)
    resnet50.summary()
