import os

import numpy as np
import tensorflow as tf
from tensorflow.python import keras

from models.networks_resnet50 import Resnet
from morphable_model.model.morphable_model import FFTfMorphableModel
from training.config_util import EasyDict
from training.train_3dmm import train_3dmm
from training.train_3dmm_warmup import train_3dmm_warmup


class FaceNetLinear3DMM:

    def __init__(self, config_general: EasyDict, config_train_warmup: EasyDict, config_train: EasyDict):
        """

        :param config_general:
             dim_illum: 10
             dim_color: 7
             dim_tex: 199
             dim_shape: 199
             dim_exp: 29
             dim_pose: 7

             save_dir: string
             bfm_dir: string

        :param config_train_warmup:
            loss_type: 'l2' or 'l1'
            face_vgg_v2_path: string
            log_freq: int
            max_checkpoint_to_keep: int
            loss_shape_type: string, optional, default l2
            loss_pose_type: string, optional, default l2
            loss_exp_type: string, optional, default l2
            loss_color_type: string, optional, default l2
            loss_illum_type: string, optional, default l2
            loss_tex_type: string, optional, default l2
            loss_landmark_type: string, optional, default l2
            data_train_dir: /opt/data/300W_LP
            data_test_dir: /opt/data/AFLW2000
            data_mean_std: /opt/data/300W_LP_stats/stats_300W_LP.npz
            batch_size: int
            learning_rate: float
            beta_1: float

        :param config_train:
            loss_type: 'l2' or 'l1'
            max_checkpoint_to_keep: int
            batch_size: int
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

        self.save_dir = self.config_general.save_dir
        self.model_dir_warm_up = os.path.join(self.save_dir, 'warm_up', 'model')
        self.eval_dir_warm_up = os.path.join(self.save_dir, 'warm_up', 'eval')
        self.log_dir_warm_up = os.path.join(self.save_dir, 'warm_up', 'log')
        self.model_dir = os.path.join(self.save_dir, 'train', 'model')
        self.eval_dir = os.path.join(self.save_dir, 'train', 'eval')
        self.log_dir = os.path.join(self.save_dir, 'train', 'log')

        if not os.path.exists(self.model_dir_warm_up):
            os.makedirs(self.model_dir_warm_up)
        if not os.path.exists(self.eval_dir_warm_up):
            os.makedirs(self.eval_dir_warm_up)
        if not os.path.exists(self.log_dir_warm_up):
            os.makedirs(self.log_dir_warm_up)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.build()

    def build(self):
        self.resnet50 = Resnet(image_size=self.image_size)
        self.resnet50.build()
        self.dense = keras.layers.Dense(units=self.output_size_structure[-1], name='3dmm_dense')
        self.model = keras.Sequential([self.resnet50.model, self.dense], name='Face3dmm')
        self.bfm = FFTfMorphableModel(
            param_mean_var_path=self.config_train_warmup.data_mean_std,
            model_path=self.config_general.bfm_dir,
            model_type='BFM'
        )

    def __call__(self, inputs, training=False):
        output = self.model(inputs, training=training)
        x_illum = tf.expand_dims(output[:, 0: self.output_size_structure[0]], axis=1)
        x_color = tf.expand_dims(output[:, self.output_size_structure[0]: self.output_size_structure[1]], axis=1)
        x_tex = tf.expand_dims(output[:, self.output_size_structure[1]: self.output_size_structure[2]], axis=2)
        x_shape = tf.expand_dims(output[:, self.output_size_structure[2]: self.output_size_structure[3]], axis=2)
        x_exp = tf.expand_dims(output[:, self.output_size_structure[3]: self.output_size_structure[4]], axis=2)
        x_pose = tf.expand_dims(output[:, self.output_size_structure[4]: self.output_size_structure[5]], axis=1)

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
        checkpoint_dir = self.model_dir_warm_up
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=self.config_train_warmup.max_checkpoint_to_keep)
        ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print('restored from {}'.format(manager.latest_checkpoint))
        else:
            print('load face vgg2 pretrained weights for resnet50')
            self.resnet50.load_pretrained(self.config_train_warmup.face_vgg_v2_path)
        self.resnet50.freeze()

        return ckpt, manager

    def _train_3dmm_warmup(self):

        ckpt, manager = self.setup_3dmm_model()

        self.summary()

        train_3dmm_warmup(
            ckpt=ckpt,
            manager=manager,
            face_model=self,
            bfm=self.bfm,
            config=self.config_train_warmup,
            log_dir=self.log_dir_warm_up,
            eval_dir=self.eval_dir_warm_up
        )

    def train(self):
        self._train_3dmm_warmup()
        self._train_3dmm()

    def _train_3dmm(self):
        self.resnet50.unfreeze()
        # we load face_vgg2 for computing loss for unsupervised learning
        # face_vgg2 will not be trainable
        self.face_vgg2 = Resnet(image_size=self.image_size)
        self.face_vgg2.build()
        self.face_vgg2.freeze()

        ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=self.model)
        checkpoint_dir = self.model_dir
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=self.config_train.max_checkpoint_to_keep)

        self.summary()

        train_3dmm(
            ckpt=ckpt,
            manager=manager,
            face_model=self,
            bfm=self.bfm,
            config=self.config_train,
            log_dir=self.log_dir,
            eval_dir=self.eval_dir
        )
