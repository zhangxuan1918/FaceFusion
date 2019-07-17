from tensorflow.python import keras
import tensorflow as tf
weight_decay = 1e-4


class IdentityBlock(keras.layers.Layer):

    def __init__(self, kernel_size: int, filters: list, stage: int, block: int, stride=1, trainable=True):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        bn_axis = 3
        self.stride = stride

        conv_name_1 = 'conv{stage}_{block}_1x1_reduce'.format(stage=stage, block=block)
        bn_name_1 = 'conv{stage}_{block}_1x1_reduce/bn'.format(stage=stage, block=block)
        self.conv1 = keras.layers.Conv2D(
            filters1,
            kernel_size=(1, 1),
            strides=(stride, stride),
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_1
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_1
        )
        self.act1 = keras.layers.Activation('relu')

        conv_name_2 = 'conv{stage}_{block}_3x3'.format(stage=stage, block=block)
        bn_name_2 = 'conv{stage}_{block}_3x3/bn'.format(stage=stage, block=block)

        self.conv2 = keras.layers.Conv2D(
            filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_2
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_2
        )
        self.act2 = keras.layers.Activation('relu')

        conv_name_3 = 'conv{stage}_{block}_1x1_increase'.format(stage=stage, block=block)
        bn_name_3 = 'conv{stage}_{block}_1x1_increase/bn'.format(stage=stage, block=block)

        self.conv3 = keras.layers.Conv2D(
            filters3,
            kernel_size=(1, 1),
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_3
        )
        self.bn3 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_3
        )
        self.act3 = keras.layers.Activation('relu')

        self.add = keras.layers.Add()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, inputs])
        x = self.act3(x)
        return x


class ConvBlock(keras.layers.Layer):
    def __init__(self, kernel_size: int, filters: list, stage: int, block: int, stride=2, trainable=True):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        bn_axis = 3
        self.stride = stride

        conv_name_1 = 'conv{stage}_{block}_1x1_reduce'.format(stage=stage, block=block)
        bn_name_1 = 'conv{stage}_{block}_1x1_reduce/bn'.format(stage=stage, block=block)

        self.conv1 = keras.layers.Conv2D(
            filters1,
            kernel_size=(1, 1),
            strides=(stride, stride),
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_1
        )
        self.bn1 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_1
        )
        self.act1 = keras.layers.Activation('relu')

        conv_name_2 = 'conv{stage}_{block}_3x3'.format(stage=stage, block=block)
        bn_name_2 = 'conv{stage}_{block}_3x3/bn'.format(stage=stage, block=block)

        self.conv2 = keras.layers.Conv2D(
            filters2,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_2
        )
        self.bn2 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_2
        )
        self.act2 = keras.layers.Activation('relu')

        conv_name_3 = 'conv{stage}_{block}_1x1_increase'.format(stage=stage, block=block)
        bn_name_3 = 'conv{stage}_{block}_1x1_increase/bn'.format(stage=stage, block=block)

        self.conv3 = keras.layers.Conv2D(
            filters3,
            kernel_size=(1, 1),
            kernel_initializer='he_normal',
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_3
        )
        self.bn3 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_3
        )

        conv_name_4 = 'conv{stage}_{block}_1x1_proj'.format(stage=stage, block=block)
        bn_name_4 = 'conv{stage}_{block}_1x1_proj/bn'.format(stage=stage, block=block)

        self.conv4 = keras.layers.Conv2D(
            filters3,
            kernel_size=(1, 1),
            strides=(stride, stride),
            use_bias=False,
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            trainable=trainable,
            name=conv_name_4
        )
        self.bn4 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name=bn_name_4
        )
        self.act3 = keras.layers.Activation('relu')
        self.add = keras.layers.Add()

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        shortcut = self.conv4(inputs)
        shortcut = self.bn4(shortcut)

        x = self.add([x, shortcut])
        x = self.act3(x)

        return x


class Resnet50(keras.Model):

    def __init__(self):
        super(Resnet50, self).__init__()
        bn_axis = 3
        self.conv1 = keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            kernel_initializer='he_normal',
            use_bias=False,
            trainable=True,
            padding='same',
            name='conv1/7x7_s2'
        )

        self.bn1 = keras.layers.BatchNormalization(
            axis=bn_axis,
            name='conv1/7x7_s2/bn'
        )
        self.act1 = keras.layers.Activation('relu')
        self.max_pool1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_bock2l = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=1, stride=1,
                                     trainable=True)
        self.identity_block22 = IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=2, trainable=True)
        self.identity_block23 = IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=3, trainable=True)

        self.conv_bock3l = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=1, stride=2, trainable=True)
        self.identity_block32 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=2, trainable=True)
        self.identity_block33 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=3, trainable=True)
        self.identity_block34 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=4, trainable=True)

        self.conv_bock4l = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=1, stride=2, trainable=True)
        self.identity_block42 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=2, trainable=True)
        self.identity_block43 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=3, trainable=True)
        self.identity_block44 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=4, trainable=True)
        self.identity_block45 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=5, trainable=True)
        self.identity_block46 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=6, trainable=True)

        self.conv_bock5l = ConvBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=1, stride=2, trainable=True)
        self.identity_block52 = IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=2, trainable=True)
        self.identity_block53 = IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=3, trainable=True)

        self.avg_pool = keras.layers.AveragePooling2D(
            pool_size=(7, 7),
            name='avg_pool'

        )
        self.flatten = keras.layers.Flatten()
        self.dim_proj = keras.layers.Dense(units=512, name='dim_proj')

    def call(self, inputs, training=True):
        # input size 224 x 224 x 3
        x = self.conv1(inputs)
        # input size 112 x 112 x 64
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.max_pool1(x)

        # input size 56 x 56
        x = self.conv_bock2l(x, training=training)
        x = self.identity_block22(x, training=training)
        x = self.identity_block23(x, training=training)

        # input size 28 x 28
        x = self.conv_bock3l(x, training=training)
        x = self.identity_block32(x, training=training)
        x = self.identity_block33(x, training=training)
        x = self.identity_block34(x, training=training)

        # input size 14 x 14
        x = self.conv_bock4l(x, training=training)
        x = self.identity_block42(x, training=training)
        x = self.identity_block43(x, training=training)
        x = self.identity_block44(x, training=training)
        x = self.identity_block45(x, training=training)
        x = self.identity_block46(x, training=training)

        # input size 7 x 7
        x = self.conv_bock5l(x, training=training)
        x = self.identity_block52(x, training=training)
        x = self.identity_block53(x, training=training)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dim_proj(x)

        return x
