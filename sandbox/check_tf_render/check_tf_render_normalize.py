import numpy as np
import scipy.io as sio
import tensorflow as tf
from tf_3dmm.mesh.render import render

from morphable_model.model.morphable_model import FFTfMorphableModel

bfm = FFTfMorphableModel(param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz', model_path='/opt/data/BFM/BFM.mat')

# --load mesh data
pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'
mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)

sp = np.divide(np.subtract(mat_data['Shape_Para'], bfm.stats_shape_mu.numpy()), bfm.stats_shape_std.numpy())
ep = np.divide(np.subtract(mat_data['Exp_Para'], bfm.stats_exp_mu.numpy()), bfm.stats_exp_std.numpy())
tp = np.divide(np.subtract(mat_data['Tex_Para'], bfm.stats_tex_mu.numpy()), bfm.stats_tex_std.numpy())
cp = np.divide(np.subtract(mat_data['Color_Para'], bfm.stats_color_mu.numpy()), bfm.stats_color_std.numpy())
ip = np.divide(np.subtract(mat_data['Illum_Para'], bfm.stats_illum_mu.numpy()), bfm.stats_illum_std.numpy())
pp = mat_data['Pose_Para']
pp[0, 3:] = pp[0, 3:] * 224 / 450
pp = np.divide(np.subtract(pp, bfm.stats_pose_mu.numpy()), bfm.stats_pose_std.numpy())

sp = tf.constant(sp, dtype=tf.float32)
ep = tf.constant(ep, dtype=tf.float32)

vertices = bfm.get_vertices(sp * bfm.stats_shape_std + bfm.stats_shape_mu,
                            ep * bfm.stats_exp_std + bfm.stats_exp_mu)
triangles = bfm.triangles

tp = tf.constant(tp, dtype=tf.float32)
cp = tf.constant(cp, dtype=tf.float32)
ip = tf.constant(ip, dtype=tf.float32)
pp = tf.constant(pp, dtype=tf.float32)

image = render(
    pose_param=pp * bfm.stats_pose_std + bfm.stats_pose_mu,
    shape_param=sp * bfm.stats_shape_std + bfm.stats_shape_mu,
    exp_param=ep * bfm.stats_exp_std + bfm.stats_exp_mu,
    tex_param=tp * bfm.stats_tex_std + bfm.stats_tex_mu,
    color_param=cp * bfm.stats_color_std + bfm.stats_color_mu,
    illum_param=ip * bfm.stats_illum_std + bfm.stats_illum_mu,
    frame_height=224,
    frame_width=224,
    tf_bfm=bfm
)

import imageio
import numpy as np

imageio.imsave('./textured_3dmm.jpg', image.numpy().astype(np.uint8))