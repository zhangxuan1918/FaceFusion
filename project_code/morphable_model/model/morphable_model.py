import numpy as np
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
import tensorflow as tf


class FFTfMorphableModel(TfMorphableModel):

    def __init__(self, param_mean_var_path, model_path, model_type='BFM'):

        super().__init__(model_path=model_path, model_type=model_type)
        self.param_mean_var_path = param_mean_var_path
        self.load_param_mean_var()

    def load_param_mean_var(self):
        params_mean_var = np.load(self.param_mean_var_path)

        # rescale pose params
        # param mean var computed for image size = 450
        # we rescale the image size to 224
        # only scaling and t3d need to be rescaled

        params_mean_var['Pose_Para_mean'][:, 3:] = params_mean_var['Pose_Para_mean'][:, 3:] * 224. / 450.
        params_mean_var['Pose_Para_std'][:, 3:] = params_mean_var['Pose_Para_std'][:, 3:] * 224. / 450.

        self.pose_mu = tf.constant(params_mean_var['Pose_Para_mean'], dtype=tf.float32)
        self.pose_std = tf.constant(params_mean_var['Pose_Para_mean'], dtype=tf.float32)

        self.exp_mu = tf.constant(params_mean_var['Exp_Para_mean'], dtype=tf.float32)
        self.exp_std = tf.constant(params_mean_var['Exp_Para_std'], dtype=tf.float32)

        self.tex_mu = tf.constant(params_mean_var['Tex_Para_mean'], dtype=tf.float32)
        self.tex_std = tf.constant(params_mean_var['Tex_Para_std'], dtype=tf.float32)

        self.color_mu = tf.constant(params_mean_var['Color_Para_mean'], dtype=tf.float32)
        self.color_std = tf.constant(params_mean_var['Color_Para_std'], dtype=tf.float32)

        self.illum_mu = tf.constant(params_mean_var['Illum_Para_mean'], dtype=tf.float32)
        self.illum_std = tf.constant(params_mean_var['Illum_Para_std'], dtype=tf.float32)

        self.shape_mu = tf.constant(params_mean_var['Shape_Para_mean'], dtype=tf.float32)
        self.shape_std = tf.constant(params_mean_var['Shape_Para_std'], dtype=tf.float32)