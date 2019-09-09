from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.activations import elu
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, Dense, AveragePooling2D
import tensorflow as tf


class FaceEncoder:
    def __init__(self, image_size, gf_dim, gfc_dim, m_dim):
        self.image_size = image_size
        self.bn_axis = 3
        self.gf_dim = gf_dim
        self.gfc_dim = gfc_dim
        self.m_dim = m_dim

    def build(self):
        inputs = Input(shape=[self.image_size, self.image_size, 3], name='image_input')
        x = self.get_encoder(inputs=inputs, is_reuse=False, is_training=True)
        self.model = Model(inputs=inputs, outputs=x, name='FaceEncoder')

    def __call__(self, inputs, training=False):
        return self.model(inputs=inputs, training=training)

    def summary(self):
        print(self.model.summary())

    def load_pretrained(self, weights_path):
        self.model.load_weights(filepath=weights_path)

    def get_encoder(self, inputs, is_reuse=False, is_training=True):
        if not is_reuse:
            self.g_bn0_0 = BatchNormalization(axis=self.bn_axis, name='g_k_bn0_0', scale=True, fused=True)
            self.g_bn0_1 = BatchNormalization(axis=self.bn_axis, name='g_k_bn0_1', scale=True, fused=True)
            self.g_bn0_2 = BatchNormalization(axis=self.bn_axis, name='g_k_bn0_2', scale=True, fused=True)
            self.g_bn0_3 = BatchNormalization(axis=self.bn_axis, name='g_k_bn0_3', scale=True, fused=True)
            self.g_bn1_0 = BatchNormalization(axis=self.bn_axis, name='g_k_bn1_0', scale=True, fused=True)
            self.g_bn1_1 = BatchNormalization(axis=self.bn_axis, name='g_k_bn1_1', scale=True, fused=True)
            self.g_bn1_2 = BatchNormalization(axis=self.bn_axis, name='g_k_bn1_2', scale=True, fused=True)
            self.g_bn1_3 = BatchNormalization(axis=self.bn_axis, name='g_k_bn1_3', scale=True, fused=True)
            self.g_bn2_0 = BatchNormalization(axis=self.bn_axis, name='g_k_bn2_0', scale=True, fused=True)
            self.g_bn2_1 = BatchNormalization(axis=self.bn_axis, name='g_k_bn2_1', scale=True, fused=True)
            self.g_bn2_2 = BatchNormalization(axis=self.bn_axis, name='g_k_bn2_2', scale=True, fused=True)
            self.g_bn2_3 = BatchNormalization(axis=self.bn_axis, name='g_k_bn2_3', scale=True, fused=True)
            self.g_bn3_0 = BatchNormalization(axis=self.bn_axis, name='g_k_bn3_0', scale=True, fused=True)
            self.g_bn3_1 = BatchNormalization(axis=self.bn_axis, name='g_k_bn3_1', scale=True, fused=True)
            self.g_bn3_2 = BatchNormalization(axis=self.bn_axis, name='g_k_bn3_2', scale=True, fused=True)
            self.g_bn3_3 = BatchNormalization(axis=self.bn_axis, name='g_k_bn3_3', scale=True, fused=True)
            self.g_bn4_0 = BatchNormalization(axis=self.bn_axis, name='g_k_bn4_0', scale=True, fused=True)
            self.g_bn4_1 = BatchNormalization(axis=self.bn_axis, name='g_k_bn4_1', scale=True, fused=True)
            self.g_bn4_2 = BatchNormalization(axis=self.bn_axis, name='g_k_bn4_2', scale=True, fused=True)
            self.g_bn4_c = BatchNormalization(axis=self.bn_axis, name='g_h_bn4_c', scale=True, fused=True)
            self.g_bn5 = BatchNormalization(axis=self.bn_axis, name='g_k_bn5', scale=True, fused=True)
            self.g_bn5_m = BatchNormalization(axis=self.bn_axis, name='g_k_bn5_m', scale=True, fused=True)
            self.g_bn5_il = BatchNormalization(axis=self.bn_axis, name='g_k_bn5_il', scale=True, fused=True)
            self.g_bn5_shape = BatchNormalization(axis=self.bn_axis, name='g_k_bn5_shape', scale=True, fused=True)
            self.g_bn5_shape_linear = BatchNormalization(axis=self.bn_axis, name='g_k_bn5_shape_linear', scale=True, fused=True)
            self.g_bn5_tex = BatchNormalization(axis=self.bn_axis, name='g_k_bn5_tex', scale=True, fused=True)

        # inputs are of size 224 x 224 x 3
        k0_1 = elu(self.g_bn0_1(Conv2D(self.gf_dim * 1, (7, 7), (2, 2), padding='VALID', use_bias=False, name='g_k01_conv')(inputs), training=is_training))
        k0_2 = elu(self.g_bn0_2(Conv2D(self.gf_dim * 2, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k02_conv')(k0_1), training=is_training))

        k1_0 = elu(self.g_bn1_0(Conv2D(self.gf_dim * 2, (3, 3), (2, 2), padding='VALID', use_bias=False, name='g_k10_conv')(k0_2), training=is_training))
        k1_1 = elu(self.g_bn1_1(Conv2D(self.gf_dim * 2, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k11_conv')(k1_0), training=is_training))
        k1_2 = elu(self.g_bn1_2(Conv2D(self.gf_dim * 4, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k12_conv')(k1_1), training=is_training))
        k2_0 = elu(self.g_bn2_0(Conv2D(self.gf_dim * 4, (3, 3), (2, 2), padding='VALID', use_bias=False, name='g_k20_conv')(k1_2), training=is_training))
        k2_1 = elu(self.g_bn2_1(Conv2D(self.gf_dim * 3, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k21_conv')(k2_0), training=is_training))
        k2_2 = elu(self.g_bn2_2(Conv2D(self.gf_dim * 6, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k22_conv')(k2_1), training=is_training))
        k3_0 = elu(self.g_bn3_0(Conv2D(self.gf_dim * 6, (3, 3), (2, 2), padding='VALID', use_bias=False, name='g_k30_conv')(k2_2), training=is_training))
        k3_1 = elu(self.g_bn3_1(Conv2D(self.gf_dim * 4, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k31_conv')(k3_0), training=is_training))
        k3_2 = elu(self.g_bn3_2(Conv2D(self.gf_dim * 8, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k32_conv')(k3_1), training=is_training))
        k4_0 = elu(self.g_bn4_0(Conv2D(self.gf_dim * 8, (3, 3), (2, 2), padding='VALID', use_bias=False, name='g_k40_conv')(k3_2), training=is_training))
        k4_1 = elu(self.g_bn4_1(Conv2D(self.gf_dim * 5, (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k41_conv')(k4_0), training=is_training))

        # Pose
        k51_m = self.g_bn5_m(Conv2D(int(self.gfc_dim / 5), (3, 3), (1, 1), padding='VALID', use_bias=False, name='g_k5_m_conv')(k4_1), training=is_training)
        k51_shape_ = get_shape(k51_m)
        k52_m = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(k51_m)
        k52_m = tf.reshape(k52_m, [-1, int(self.gfc_dim / 5)])
        k6_m = Dense(self.m_dim, 'g_k6_m_lin')(k52_m)

        # Illumination
        k51_ill = self.g_bn5_il(Conv2D(int(self.gfc_dim / 5), (3, 3), (1, 1), padding='VALID', name='g_k5_il_conv')(k4_1), training=is_training)
        k52_ill = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(k51_ill)
        k52_ill = tf.reshape(k52_ill, [-1, int(self.gfc_dim / 5)])
        k6_ill = Dense(self.il_dim, 'g_k6_il_lin')(k52_ill)

        # Shape
        k51_shape = self.g_bn5_shape(Conv2D(self.gfc_dim / 2, (3, 3), (1, 1), padding='VALID', name='g_k5_shape_conv')(k4_1), train=is_training)
        k52_shape = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(k51_shape)
        k52_shape = tf.reshape(k52_shape, [-1, int(self.gfc_dim / 2)])
        k6_shape = Dense(self.sh_dim, 'g_k6_sh_lin')(k52_shape)

        # Texture
        k51_tex = self.g_bn5_tex(Conv2D(self.gfc_dim / 2, (3, 3), (1, 1), padding='VALID', name='g_k5_tex_conv')(k4_1),
                                 training=is_training)
        k52_tex = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(
            k51_tex)
        k52_tex = tf.reshape(k52_tex, [-1, int(self.gfc_dim / 2)])
        k6_tex = Dense(self.tx_dim, 'g_k6_tx_lin')(k52_tex)

        # Expression
        k51_exp = self.g_bn5_exp(Conv2D(self.gfc_dim / 5, (3, 3), (1, 1), padding='VALID', name='g_k5_exp_conv')(k4_1), train=is_training)
        k52_exp = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(k51_exp)
        k52_exp = tf.reshape(k52_exp, [-1, int(self.gfc_dim / 2)])
        k6_exp = Dense(self.sh_dim, 'g_k6_ex_lin')(k52_exp)

        # Color
        k51_col = self.g_bn5_col(Conv2D(self.gfc_dim / 5, (3, 3), (1, 1), padding='VALID', name='g_k5_col_conv')(k4_1), train=is_training)
        k52_col = AveragePooling2D(ksize=[1, k51_shape_[1], k51_shape_[2], 1], strides=[1, 1, 1, 1], padding='VALID')(k51_col)
        k52_col = tf.reshape(k52_col, [-1, int(self.gfc_dim / 2)])
        k6_col = Dense(self.co_dim, 'g_k6_ex_lin')(k52_col)

        return k6_shape, k6_tex, k6_exp, k6_m, k6_ill, k6_col


if __name__ == '__main__':
    resnet50 = Resnet(image_size=224)
    resnet50.build()
    resnet50.load_pretrained(weights_path='/opt/project/project_code/data/pretrained_model/20190718/weights.h5')
    resnet50.summary()
