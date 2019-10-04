import numpy as np
from tf_3dmm.mesh.render import render_2
import tensorflow as tf
from morphable_model.model.morphable_model import FFTfMorphableModel
import scipy.io as sio

from training.opt import save_images

bfm = FFTfMorphableModel(param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz', model_path='/opt/data/BFM/BFM.mat')

# --load mesh data
pic_name = 'image00283'
# pic_name = 'image00278'
# pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'

jpg_filename = '../../examples/Data/{0}.jpg'.format(pic_name)
image = tf.io.read_file(jpg_filename)
image = tf.image.decode_jpeg(image, channels=3)

mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)

tx = 32
ty = 17
render_image_size = 224

image_resized = tf.image.resize(image, (256, 256))
image_shift = tf.image.crop_to_bounding_box(image_resized, tx, ty, render_image_size, render_image_size)
image_resized2 = tf.image.resize(image, (224, 224))

lm = mat_data['pt3d_68'][0:2, :]
lm_shift = np.copy(lm)
lm_shift *= 256. / 450.
lm_shift[0], lm_shift[1] = lm_shift[0] - ty, lm_shift[1] - tx
lm *= 224. / 450

sp = mat_data['Shape_Para']
ep = mat_data['Exp_Para']
tp = mat_data['Tex_Para'][:40, :]
cp = mat_data['Color_Para']
ip = mat_data['Illum_Para']
pp = mat_data['Pose_Para']
pp[0, 3:5] = pp[0, 3:5] * 256 / 450
pp[0, 6] = pp[0, 6] * 256 / 450
pp[0, 3], pp[0, 4] = pp[0, 3] - ty, pp[0, 4] - (32 - tx)


shape_param = tf.constant(sp, dtype=tf.float32)
exp_param = tf.constant(ep, dtype=tf.float32)
tex_param = tf.constant(tp, dtype=tf.float32)
color_param = tf.constant(cp, dtype=tf.float32)
illum_param = tf.constant(ip, dtype=tf.float32)
pose_param = tf.constant(pp, dtype=tf.float32)

image_rendered = render_2(
    angles_grad=pose_param[0, 0:3],
    t3d=pose_param[0, 3:6],
    scaling=pose_param[0, 6],
    shape_param=shape_param,
    exp_param=exp_param,
    tex_param=tex_param,
    color_param=color_param,
    illum_param=illum_param,
    frame_height=render_image_size,
    frame_width=render_image_size,
    tf_bfm=bfm
)

save_to_file = './test_landmark_shift2.jpg'


save_images(
    images=[image_resized2.numpy().astype(np.uint8), image_shift.numpy().astype(np.uint8), image_rendered.numpy().astype(np.uint8)],
    landmarks=[lm, lm_shift, lm_shift],
    titles=['orignal', 'shifted', 'rendered'],
    filename=save_to_file
)