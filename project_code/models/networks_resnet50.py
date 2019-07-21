from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.keras import layers, Input, Model
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense
from tensorflow.python.keras.regularizers import l2

global weight_decay
weight_decay = 1e-4


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay),
                      trainable=trainable,
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


class Resnet:

    def __init__(self, image_size):
        self.image_size = image_size

    def build(self):
        inputs = Input(shape=[self.image_size, self.image_size, 3], name='image_input')
        x = self.resnet50_backend(inputs=inputs)

        # AvgPooling
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='dim_proj')(x)

        self.model = Model(inputs=inputs, outputs=x, name='Resnet50')

    def summary(self):
        print(self.model.summary())

    def freeze(self):
        self.model.trainable = False

    def unfreeze(self):
        self.model.trainable = True

    def load_pretrained(self, weights_path):
        self.model.load_weights(filepath=weights_path)

    def resnet50_backend(self, inputs):
        bn_axis = 3
        # inputs are of size 224 x 224 x 3
        x = Conv2D(64, (7, 7), strides=(2, 2),
                   kernel_initializer='orthogonal',
                   use_bias=False,
                   trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   padding='same',
                   name='conv1/7x7_s2')(inputs)

        # inputs are of size 112 x 112 x 64
        x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # inputs are of size 56 x 56
        x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block=2, trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block=3, trainable=True)

        # inputs are of size 28 x 28
        x = conv_block(x, 3, [128, 128, 512], stage=3, block=1, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=2, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=3, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=4, trainable=True)

        # inputs are of size 14 x 14
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6, trainable=True)

        # inputs are of size 7 x 7
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1, trainable=True)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2, trainable=True)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3, trainable=True)
        return x

    def resnet50_backend_truncated(self, inputs):
        bn_axis = 3
        # inputs are of size 224 x 224 x 3
        x = Conv2D(64, (7, 7), strides=(2, 2),
                   kernel_initializer='orthogonal',
                   use_bias=False,
                   trainable=True,
                   kernel_regularizer=l2(weight_decay),
                   padding='same',
                   name='conv1/7x7_s2')(inputs)

        # inputs are of size 112 x 112 x 64
        x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # inputs are of size 56 x 56
        x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block=2, trainable=True)
        x = identity_block(x, 3, [64, 64, 256], stage=2, block=3, trainable=True)

        # inputs are of size 28 x 28
        x = conv_block(x, 3, [128, 128, 512], stage=3, block=1, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=2, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=3, trainable=True)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block=4, trainable=True)

        # inputs are of size 14 x 14
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5, trainable=True)
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6, trainable=True)
        return x


if __name__ == '__main__':
    resnet50 = Resnet(image_size=224)
    resnet50.build()
    resnet50.load_pretrained(weights_path='/opt/project/project_code/data/pretrained_model/20190718/weights.h5')
    resnet50.summary()
