import os

import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.models.networks_resnet50 import Resnet50, resnet50_backend, Resnet


class FaceNetLinear3DMM:

    def __init__(self, config_general, config_train_warmup, config_train):
        """

        :param config_general:
             is_using_warmup: boolean
             dim_illum: 10
             dim_color: 7
             dim_tex: 199
             dim_shape: 199
             dim_exp: 29
             dim_pose: 7

             checkout_dir: string
             bfm_dir

        :param config_train_warmup:
            loss_type: 'l2' or 'l1'
            face_vgg_v2_path: string
            log_freq: int
            loss_shape_type: string, optional, default l2
            loss_pose_type: string, optional, default l2
            loss_exp_type: string, optional, default l2
            loss_color_type: string, optional, default l2
            loss_illum_type: string, optional, default l2
            loss_tex_type: string, optional, default l2
            loss_landmark_type: string, optional, default l2

        :param config_train:
            loss_type: 'l2' or 'l1'
        """
        self.config_general = config_general
        self.config_train_warmup = config_train_warmup
        self.config_train = config_train

        self.image_size = 224

        self.output_size_structure = list(np.cumsum([
            self.config_general.dim_illum,
            self.config_general.dim_color,
            self.config_general.dim_tex,
            self.config_general.dim_shape,
            self.config_general.dim_exp,
            self.config_general.dim_pose
        ]))

        self.checkpoint_dir = self.config_general.checkout_dir
        self.save_dir = self.config_general.save_dir
        self.model_dir_warm_up = os.path.join(self.save_dir, 'model_warm_up')
        self.eval_dir_warm_up = os.path.join(self.save_dir, 'eval_warm_up')
        self.log_dir_warm_up = os.path.join(self.save_dir, 'log_warm_up')
        self.model_dir = os.path.join(self.save_dir, 'model')
        self.eval_dir = os.path.join(self.save_dir, 'eval')
        self.log_dir = os.path.join(self.save_dir, 'log')

        if not os.path.exists(self.model_dir_warm_up):
            os.makedirs(self.model_dir_warm_up)
        if not os.path.exists(self.eval_dir_warm_up):
            os.makedirs(self.eval_dir_warm_up)
        if not os.path.exists(self.model_dir_warm_up):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.eval_dir_warm_up):
            os.makedirs(self.eval_dir)
        if not os.path.exists(self.log_dir_warm_up):
            os.makedirs(self.log_dir_warm_up)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build()

    def build(self):
        self.resnet50 = Resnet(image_size=self.image_size)
        self.dense = keras.layers.Dense(units=self.output_size_structure[-1], name='3dmm_dense')
        self.model = keras.Sequential([self.resnet50.model, self.dense])
        self.bfm = TfMorphableModel(model_path=self.config_general.bfm_dir)

    def __call__(self, inputs, training=True):
        output = self.model(inputs)
        x_illum = tf.expand_dims(output[:, 0: self.output_size_structure[1]], axis=1)
        x_color = tf.expand_dims(output[:, self.output_size_structure[1]: self.output_size_structure[2]], axis=1)
        x_tex = tf.expand_dims(output[:, self.output_size_structure[2]: self.output_size_structure[3]], axis=2)
        x_shape = tf.expand_dims(output[:, self.output_size_structure[3]: self.output_size_structure[4]], axis=2)
        x_exp = tf.expand_dims(output[:, self.output_size_structure[4]: self.output_size_structure[5]], axis=2)
        x_pose = tf.expand_dims(output[:, self.output_size_structure[5]: self.output_size_structure[6]], axis=1)

        return {
            'illum': x_illum,
            'color': x_color,
            'tex': x_tex,
            'shape': x_shape,
            'exp': x_exp,
            'pose': x_pose
        }

    def summary(self):
        print(self.model.summary())

    def setup_3dmm_model(self):
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
        manager = tf.train.CheckpointManager(ckpt, self.checkpoint_dir, max_to_keep=self.config_general.max_checkpoint_to_keep)
        ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print('restored from {}'.format(manager.latest_checkpoint))
        else:
            print('load face vgg2 pretrained weights for resnet50')
            self.resnet50.load_pretrained(self.config_train_warmup.face_vgg_v2_path)
        self.resnet50.freeze()

        return ckpt, manager

    def _train_3dmm_warmup(self, numof_epochs):

        ckpt, manager = self.setup_3dmm_model()

        self.summary()

        train_3dmm_warmup(
            numof_epochs=numof_epochs,
            ckpt=ckpt,
            manager=manager,
            face_model=self,
            bfm=self.bfm,
            config=self.config_train_warmup,
            log_dir=self.log_dir_warm_up,
            eval_dir=self.eval_dir_warm_up
        )

    def train(self, numof_epochs_warmup, numof_epochs):
        if self.config_general.is_using_warmup:
            # use supervised learning
            self._train_3dmm_warmup(numof_epochs_warmup)

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

        self.resnet50 = Resnet(batch_size=self.batch_size, )

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
