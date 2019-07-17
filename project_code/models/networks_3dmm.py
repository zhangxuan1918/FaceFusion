import numpy as np
from tensorflow.python import keras
import tensorflow as tf
from project_code.models.networks_resnet50 import Resnet50


class Face3DMM(keras.Model):

    def __init__(self,
                 size_illum_param: int = 10,
                 size_color_param: int = 7,
                 size_tex_param: int = 199,
                 size_shape_param: int = 199,
                 size_exp_param: int = 29,
                 size_pose_param: int = 7
                 ):
        super().__init__()
        self.resnet = Resnet50()
        self.size_illum_param = size_illum_param
        self.size_color_param = size_color_param
        self.size_tex_param = size_tex_param
        self.size_shape_param = size_shape_param
        self.size_exp_param = size_exp_param
        self.size_pose_param = size_pose_param

        self.output_meta = list(np.cumsum([
            self.size_illum_param,
            self.size_color_param,
            self.size_tex_param,
            self.size_shape_param,
            self.size_exp_param,
            self.size_pose_param
        ]))
        self.dense = keras.layers.Dense(units=self.output_meta[-1], name='3dmm_dense')

    def freeze_resnet(self):
        self.resnet.trainable = False

    def unfreeze_resnet(self):
        self.resnet.trainable = True

    def call(self, inputs, training=True):
        x = self.resnet(inputs=inputs, training=training)

        x = self.dense(x)

        return self.split_output(x)

    def split_output(self, output):
        x_illum = tf.expand_dims(output[0: self.output_meta[1]], axis=0)
        x_color = tf.expand_dims(output[self.output_meta[1]: self.output_meta[2]], axis=0)
        x_tex = tf.expand_dims(output[self.output_meta[2]: self.output_meta[3]], axis=1)
        x_shape = tf.expand_dims(output[self.output_meta[3]: self.output_meta[4]], axis=1)
        x_exp = tf.expand_dims(output[self.output_meta[4]: self.output_meta[5]], axis=1)
        x_pose = tf.expand_dims(output[self.output_meta[5]: self.output_meta[6]], axis=0)

        return x_illum, x_color, x_tex, x_shape, x_exp, x_pose
