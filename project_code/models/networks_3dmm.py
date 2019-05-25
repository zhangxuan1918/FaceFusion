from project_code.models.networks_head import Head3dmm
from project_code.models.networks_resnet50 import Resnet50


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

        self.head_landmark = Head3dmm(2 * self.size_landmark, header_name='head_landmark')
        self.head_illum = Head3dmm(self.size_illum_param, header_name='head_illum')
        self.head_color = Head3dmm(self.size_color_param, header_name='head_color')
        self.head_tex = Head3dmm(self.size_tex_param, header_name='head_tex')
        self.head_shape = Head3dmm(self.size_shape_param, header_name='head_shape')
        self.head_exp = Head3dmm(self.size_exp_param, header_name='head_exp')
        self.head_pose = Head3dmm(self.size_pose_param, header_name='head_pose')

    def call(self, inputs, training=True):
        x = super(Face3DMM, self).call(inputs=inputs, training=training)

        x_landmark = self.head_landmark(x)
        x_illum = self.head_illum(x)
        x_color = self.head_color(x)
        x_tex = self.head_tex(x)
        x_shape = self.head_shape(x)
        x_exp = self.head_exp(x)
        x_pose = self.head_pose(x)

        return [x_landmark, x_illum, x_color, x_tex, x_shape, x_exp, x_pose]
