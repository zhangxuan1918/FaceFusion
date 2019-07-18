import os

import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from project_code.models.networks_resnet50 import Resnet50, resnet50_backend


class FaceNetLinear3DMM:

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.image_size = 224
        self.rendered_image_size = 450

        self.is_using_warmup = config.is_using_warmup

        self.loss_warmup_type = config.warmup_loss_type if hasattr(config, 'warmup_loss_type') else 'l2'
        self.loss_recon_type = config.recon_loss_type if hasattr(config, 'recon_loss_type') else 'l2'

        self.dim_illum = config.size_illum_param if hasattr(config, 'dim_illum') else 10
        self.dim_color = config.size_color_param if hasattr(config, 'dim_color') else 7
        self.dim_tex = config.size_tex_param if hasattr(config, 'dim_tex') else 199
        self.dim_shape = config.size_shape_param if hasattr(config, 'dim_shape') else 199
        self.dim_exp = config.size_exp_param if hasattr(config, 'dim_exp') else 29
        self.dim_pose = config.size_pose_param if hasattr(config, 'dim_pose') else 7

        self.output_size_structure = list(np.cumsum([
            self.dim_illum,
            self.dim_color,
            self.dim_tex,
            self.dim_shape,
            self.dim_exp,
            self.dim_pose
        ]))

        self.checkpoint_dir = config.checkout_dir
        self.resnet_weights_dir = config.resnet_weights_dir
        self.save_dir = config.save_dir
        self.model_dir = os.path.join(self.save_dir, 'model')
        self.eval_dir = os.path.join(self.save_dir, 'eval')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)

        self.build()

    def build(self):
        self.resnet = resnet50_backend(tf.zeros([None, self.image_size, self.image_size, 3]))
        self.dense = keras.layers.Dense(units=self.output_size_structure[-1], name='3dmm_dense')
        self.model = keras.Sequential([self.resnet, self.dense])

    def summary(self):
        print(self.model.summary())

    def setup_3dmm_data(self):
        pass

    def _train_3dmm_warmup(self):
        train_ds, test_ds = self.setup_3dmm_data()
        self._load_resnet_weights()

    def train(self):
        if self.is_using_warmup:
            # use supervised learning
            self._train_3dmm_warmup()

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
