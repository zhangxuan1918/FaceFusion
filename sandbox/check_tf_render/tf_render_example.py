import scipy.io as sio
import tensorflow as tf
from tf_3dmm.mesh.render import render
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

tf_bfm = TfMorphableModel('../../examples/Data/BFM/Out/BFM.mat')
# --load mesh data
pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'
mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)
sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

vertices = tf_bfm.get_vertices(sp, ep)
triangles = tf_bfm.triangles

tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

image = render(
    pose_param=pp,
    shape_param=sp,
    exp_param=ep,
    tex_param=tp,
    color_param=cp,
    illum_param=ip,
    frame_height=450,
    frame_width=450,
    tf_bfm=tf_bfm
)

import imageio
import numpy as np

imageio.imsave('./textured_3dmm.jpg', image.numpy().astype(np.uint8))