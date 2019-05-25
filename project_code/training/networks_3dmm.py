from tensorflow.python import keras

weight_decay = 1e-4


class IdentityBlock(keras.Model):

    def __init__(self, kernel_size: int, filters: list, stage: int, block: int, trainable=True):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = 'conv{stage}_{block}_1x1_reduce'.format(stage=stage, block=block)
        bn_name_1 = 'conv{stage}_{block}_1x1_reduce/bn'.format(stage=stage, block=block)

        self.conv1 = keras.layers.Conv2D(
            filters1,
            kernel_size=(1, 1),
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

    def call(self, input_tensor, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.add([x, input_tensor])
        x = self.act3(x)
        return x


class ConvBlock(keras.Model):
    def __init__(self, kernel_size: int, filters: list, stage: int, block: int, strides=(2, 2), trainable=True):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        bn_axis = 3

        conv_name_1 = 'conv{stage}_{block}_1x1_reduce'.format(stage=stage, block=block)
        bn_name_1 = 'conv{stage}_{block}_1x1_reduce/bn'.format(stage=stage, block=block)

        self.conv1 = keras.layers.Conv2D(
            filters1,
            kernel_size=(1, 1),
            strides=strides,
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
            strides=strides,
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

    def call(self, input_tensor, **kwargs):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv4(input_tensor)
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

        self.conv_bock2l = ConvBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=1, strides=(1, 1),
                                     trainable=True)
        self.identity_block22 = IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=2, trainable=True)
        self.identity_block23 = IdentityBlock(kernel_size=3, filters=[64, 64, 256], stage=2, block=3, trainable=True)

        self.conv_bock3l = ConvBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=1, trainable=True)
        self.identity_block32 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=2, trainable=True)
        self.identity_block33 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=3, trainable=True)
        self.identity_block34 = IdentityBlock(kernel_size=3, filters=[128, 128, 512], stage=3, block=4, trainable=True)

        self.conv_bock4l = ConvBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=1, trainable=True)
        self.identity_block42 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=2, trainable=True)
        self.identity_block43 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=3, trainable=True)
        self.identity_block44 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=4, trainable=True)
        self.identity_block45 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=5, trainable=True)
        self.identity_block46 = IdentityBlock(kernel_size=3, filters=[256, 256, 1024], stage=4, block=6, trainable=True)

        self.conv_bock5l = ConvBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=1, trainable=True)
        self.identity_block52 = IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=2, trainable=True)
        self.identity_block53 = IdentityBlock(kernel_size=3, filters=[512, 512, 2048], stage=5, block=3, trainable=True)

    def call(self, input_tensor, **kwargs):
        # input size 224 x 224 x 3
        x = self.conv1(input_tensor)
        # input size 112 x 112 x 64
        x = self.bn1(x)
        x = self.act1(x)
        x = self.max_pool1(x)

        # input size 56 x 56
        x = self.conv_bock2l(x)
        x = self.identity_block22(x)
        x = self.identity_block23(x)

        # input size 28 x 28
        x = self.conv_bock3l(x)
        x = self.identity_block32(x)
        x = self.identity_block33(x)
        x = self.identity_block34(x)

        # input size 14 x 14
        x = self.conv_bock4l(x)
        x = self.identity_block42(x)
        x = self.identity_block43(x)
        x = self.identity_block44(x)
        x = self.identity_block45(x)
        x = self.identity_block46(x)

        # input size 7 x 7
        x = self.conv_bock5l(x)
        x = self.identity_block52(x)
        x = self.identity_block53(x)

        return x


class Face3DMM(Resnet50):

    def __init__(self, size_landmark: int = 68,
                 size_illum_param: int = 10,
                 size_color_param: int = 7,
                 size_tex_param: int = 199,
                 size_shape_param: int = 199,
                 size_exp_param: int = 29,
                 size_pose_param: int = 7,
                 ):
        super(Face3DMM, self).__init__()
        self.size_landmark = size_landmark
        self.size_illum_param = size_illum_param
        self.size_color_param = size_color_param
        self.size_tex_param = size_tex_param
        self.size_shape_param = size_shape_param
        self.size_exp_param = size_exp_param
        self.size_pose_param = size_pose_param

        self.dense = keras.layers.Dense(units=512, activation='relu', use_bias=True, name='dim_proj')

        self.head_landmark = keras.layers.Dense(2 * self.size_landmark, activation=None, use_bias=True,
                                                name='head_landmark')
        self.head_illum = keras.layers.Dense(self.size_illum_param, activation=None, use_bias=True,
                                             name='head_illum')
        self.head_color = keras.layers.Dense(self.size_color_param, activation=None, use_bias=True,
                                             name='head_color')
        self.head_tex = keras.layers.Dense(self.size_tex_param, activation=None, use_bias=True,
                                           name='head_tex')
        self.head_shape = keras.layers.Dense(self.size_shape_param, activation=None, use_bias=True,
                                             name='head_shape')
        self.head_exp = keras.layers.Dense(self.size_exp_param, activation=None, use_bias=True,
                                           name='head_exp')
        self.head_pose = keras.layers.Dense(self.size_pose_param, activation=None, use_bias=True,
                                            name='head_pose')

    def call(self, input_tensor, **kwargs):
        x = super(Face3DMM, self).call(input_tensor=input_tensor)

        x = self.dense(x)

        x_landmark = self.head_landmark(x)
        x_illum = self.head_illum(x)
        x_color = self.head_color(x)
        x_tex = self.head_tex(x)
        x_shape = self.head_shape(x)
        x_exp = self.head_exp(x)
        x_pose = self.head_pose(x)

        return [x_landmark, x_illum, x_color, x_tex, x_shape, x_exp, x_pose]