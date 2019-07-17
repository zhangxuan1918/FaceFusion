from tensorflow.python import keras
import tensorflow as tf
from project_code.models.networks_resnet50 import Resnet50, resnet50_backend


def facenet(inputs, backend='resnet50'):

    if backend == 'linear':
        return facenet_linear_3dmm(inputs=inputs)
    elif backend == 'nonlinear':
        return facenet_nonlinear_3dmm(inputs=inputs)
    else:
        raise Exception('encoding network not supported: {0}'.format(encoding))


def facenet_linear_3dmm(inputs, pretrained_model=None, trunk_trainable=False):
    trunk = resnet50_backend(inputs)

    if pretrained_model is not None:
        # TODO load pretrained model
        pass

    if not trunk_trainable:
        trunk.trainable = False

    # add headers for 3dmm model
    head_illum = keras.layers.Dense(units=10, name='head_illum')
    head_color = keras.layers.Dense(units=self.size_color_param, name='head_color')
    head_tex = keras.layers.Dense(units=self.size_tex_param, name='head_tex')
    head_shape = keras.layers.Dense(units=self.size_shape_param, name='head_shape')
    head_exp = keras.layers.Dense(units=self.size_exp_param, name='head_exp')
    head_pose = keras.layers.Dense(units=self.size_pose_param, name='head_pose')


def facenet_nonlinear_3dmm(inputs):
    raise Exception('nonlinear facenet is not implemented yet')


class Face3DMMLinear:

    def __init__(self,
                 size_illum_param: int = 10,
                 size_color_param: int = 7,
                 size_tex_param: int = 199,
                 size_shape_param: int = 199,
                 size_exp_param: int = 29,
                 size_pose_param: int = 7
                 ):
        super().__init__()
        self.size_illum_param = size_illum_param
        self.size_color_param = size_color_param
        self.size_tex_param = size_tex_param
        self.size_shape_param = size_shape_param
        self.size_exp_param = size_exp_param
        self.size_pose_param = size_pose_param

        self.trunk = None
        self.head_illum = None
        self.head_color = None
        self.head_tex = None
        self.head_shape = None
        self.head_exp = None
        self.head_pose = None
        self.model = None

    def build(self, inputs):

        self.trunk = resnet50_backend(inputs=inputs)

        self.head_illum = keras.layers.Dense(units=self.size_illum_param, name='head_illum')
        self.head_color = keras.layers.Dense(units=self.size_color_param, name='head_color')
        self.head_tex = keras.layers.Dense(units=self.size_tex_param, name='head_tex')
        self.head_shape = keras.layers.Dense(units=self.size_shape_param, name='head_shape')
        self.head_exp = keras.layers.Dense(units=self.size_exp_param, name='head_exp')
        self.head_pose = keras.layers.Dense(units=self.size_pose_param, name='head_pose')
        self.mode = tf.keras.Sequential([
                      trunk,
                      []
                    ])

    def freeze_trunk(self):
        self.trunk.trainable = False

    def unfreeze_trunk(self):
        self.trunk.trainable = True

    def call(self, inputs, training=True):
        x = self.resnet(inputs=inputs, training=training)

        x_illum = self.head_illum(x)
        x_color = self.head_color(x)
        x_tex = self.head_tex(x)
        x_shape = self.head_shape(x)
        x_exp = self.head_exp(x)
        x_pose = self.head_pose(x)

        return [x_illum, x_color, x_tex, x_shape, x_exp, x_pose]
