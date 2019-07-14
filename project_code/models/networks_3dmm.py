from tensorflow.python import keras

from project_code.models.networks_resnet50 import Resnet50


class Face3DMM(keras.Model):

    def __init__(self,
                 size_illum_param: int = 10,
                 size_color_param: int = 7,
                 size_tex_param: int = 199,
                 size_shape_param: int = 199,
                 size_exp_param: int = 29,
                 size_pose_param: int = 7,
                 illum_loss_type: str = 'l2',
                 color_loss_type: str = 'l1',
                 tex_loss_type: str = 'l1',
                 shape_loss_type: str = 'l2',
                 exp_loss_type: str = 'l2',
                 pose_loss_type: str = 'l2',
                 ):
        super().__init__()
        self.resnet = Resnet50()
        self.size_illum_param = size_illum_param
        self.size_color_param = size_color_param
        self.size_tex_param = size_tex_param
        self.size_shape_param = size_shape_param
        self.size_exp_param = size_exp_param
        self.size_pose_param = size_pose_param

        self.illum_loss_type = illum_loss_type
        self.color_loss_type = color_loss_type
        self.tex_loss_type = tex_loss_type
        self.shape_loss_type = shape_loss_type
        self.exp_loss_type = exp_loss_type
        self.pose_loss_type = pose_loss_type

        self.head_illum = keras.layers.Dense(units=self.size_illum_param, name='head_illum')
        self.head_color = keras.layers.Dense(units=self.size_color_param, name='head_color')
        self.head_tex = keras.layers.Dense(units=self.size_tex_param, name='head_tex')
        self.head_shape = keras.layers.Dense(units=self.size_shape_param, name='head_shape')
        self.head_exp = keras.layers.Dense(units=self.size_exp_param, name='head_exp')
        self.head_pose = keras.layers.Dense(units=self.size_pose_param, name='head_pose')

    def freeze_resnet(self):
        self.resnet.trainable = False

    def get_illum_trainable_vars(self):
        return self.head_illum.trainable_variables

    def get_illum_loss_type(self):
        return self.illum_loss_type

    def get_color_trainable_vars(self):
        return self.head_color.trainable_variables

    def get_color_loss_type(self):
        return self.color_loss_type

    def get_tex_trainable_vars(self):
        return self.head_tex.trainable_variables

    def get_tex_loss_type(self):
        return self.tex_loss_type

    def get_shape_trainable_vars(self):
        return self.head_shape.trainable_variables

    def get_shape_loss_type(self):
        return self.shape_loss_type

    def get_exp_trainable_vars(self):
        return self.head_exp.trainable_variables

    def get_exp_loss_type(self):
        return self.exp_loss_type

    def get_pose_trainable_vars(self):
        return self.head_pose.trainable_variables

    def get_pose_loss_type(self):
        return self.pose_loss_type

    def call(self, inputs, training=True):
        x = self.resnet(inputs=inputs, training=training)

        x_illum = self.head_illum(x)
        x_color = self.head_color(x)
        x_tex = self.head_tex(x)
        x_shape = self.head_shape(x)
        x_exp = self.head_exp(x)
        x_pose = self.head_pose(x)

        return [x_illum, x_color, x_tex, x_shape, x_exp, x_pose]
