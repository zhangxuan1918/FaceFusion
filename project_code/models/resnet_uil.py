import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.python.keras.regularizers import l2

WEIGHT_DECAY = 1e-4


def identity_block(input_tensor, kernel_size, filters, stride, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: integer
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_3x3_1'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_3x3_1/bn'
    x = Conv2D(filters, (3, 3),
               strides=(stride, stride),
               padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3_2'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3_2/bn'
    x = Conv2D(filters, kernel_size,
               strides=(1, 1),
               padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stride, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: filters
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """

    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_3x3_1'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_3x3_1/bn'
    x = Conv2D(filters, (3, 3),
               strides=(stride, stride),
               padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3_2'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3_2/bn'
    x = Conv2D(filters, kernel_size,
               strides=(1, 1),
               padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_3x3_proj'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_3x3_proj/bn'
    shortcut = Conv2D(filters, (1, 1),
                      strides=(stride, stride),
                      padding='same',
                      kernel_initializer=tf.initializers.glorot_normal(),
                      use_bias=False,
                      kernel_regularizer=l2(WEIGHT_DECAY),
                      trainable=trainable,
                      name=conv_name_3)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_3)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block_v2(input_tensor, kernel_size, filters, stage, block, trainable=True):
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
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_v2(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
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
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer=tf.initializers.glorot_normal(),
               use_bias=False,
               kernel_regularizer=l2(WEIGHT_DECAY),
               trainable=trainable,
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer=tf.initializers.glorot_normal(),
                      use_bias=False,
                      kernel_regularizer=l2(WEIGHT_DECAY),
                      trainable=trainable,
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_4)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
