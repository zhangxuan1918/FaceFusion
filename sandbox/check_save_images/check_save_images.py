import PIL
import numpy as np
import scipy.io as sio
import tensorflow as tf
from tf_3dmm.mesh.render import render
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.training.eval import save_images

tf_bfm = TfMorphableModel('/opt/project/examples/Data/BFM/Out/BFM.mat')
# --load mesh data
pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'
image_filename = '/opt/project/examples/Data/{0}.jpg'.format(pic_name)
with open(image_filename, 'rb') as file:
    img = PIL.Image.open(image_filename)
    img = np.asarray(img, dtype=np.int)

mat_filename = '/opt/project/examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)
sp = tf.constant(mat_data['Shape_Para'], dtype=tf.float32)
ep = tf.constant(mat_data['Exp_Para'], dtype=tf.float32)

vertices = tf_bfm.get_vertices(sp, ep)
triangles = tf_bfm.triangles

tp = tf.constant(mat_data['Tex_Para'], dtype=tf.float32)
cp = tf.constant(mat_data['Color_Para'], dtype=tf.float32)
ip = tf.constant(mat_data['Illum_Para'], dtype=tf.float32)
pp = tf.constant(mat_data['Pose_Para'], dtype=tf.float32)

landmarks = mat_data['pt2d']

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
).numpy().astype(np.uint8)

save_images(
    images=[img, image],
    titles=['test1', 'test2'],
    file_to_save='test.jpg',
    landmarks=[landmarks, landmarks]
)

