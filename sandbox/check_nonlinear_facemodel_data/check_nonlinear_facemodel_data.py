import numpy as np
import scipy.io as sio
from tf_3dmm.mesh.render import render
import tensorflow as tf
from morphable_model.model.morphable_model import FFTfMorphableModel
from training.opt import save_images

filelist_name = '/opt/data2/300W_LP/filelist/IBUG_filelist.txt'
fileparam_name = '/opt/data2/300W_LP/filelist/IBUG_param.dat'

filenames = []
with open(filelist_name) as f:
    for row in f:
        filenames.append(row)

idDim = 1
mDim = idDim + 8
poseDim = mDim + 7
shapeDim = poseDim + 199
expDim = shapeDim + 29
texDim = expDim + 40
ilDim = texDim + 10

with open(fileparam_name) as f:
    params = np.fromfile(file=f, dtype=np.float32)

params = params.reshape((-1, ilDim)).astype(np.float32)
pid = params[:, 0:idDim]
m = params[:, idDim:mDim]
pose = params[:, mDim:poseDim]
shape = params[:, poseDim:shapeDim]
exp = params[:, shapeDim:expDim]
tex = params[:, expDim:texDim]
il = params[:, texDim:ilDim]

for i in range(len(filenames)):
    if filenames[i].strip() == 'IBUG/IBUG_image_008_1_0.png':
        pose1 = pose[i]
        pose1 = np.expand_dims(pose1, 0)
        shape1 = shape[i]
        shape1 = np.expand_dims(shape1, 1)
        exp1 = exp[i]
        exp1 = np.expand_dims(exp1, 1)
        tex1 = tex[i]
        tex1 = np.expand_dims(tex1, 1)
        il1 = il[i]
        il1 = np.expand_dims(il1, 0)
        mat_filename = '../../examples/Data/IBUG_image_008_1_0.mat'
        mat_data = sio.loadmat(mat_filename)
        co1 = mat_data['Color_Para']
    elif filenames[i].strip() == 'IBUG/IBUG_image_014_01_2.png':
        pose2 = pose[i]
        pose2 = np.expand_dims(pose2, 0)
        shape2 = shape[i]
        shape2 = np.expand_dims(shape2, 1)
        exp2 = exp[i]
        exp2 = np.expand_dims(exp2, 1)
        tex2 = tex[i]
        tex2 = np.expand_dims(tex2, 1)
        il2 = il[i]
        il2 = np.expand_dims(il2, 0)
        mat_filename = '../../examples/Data/IBUG_image_014_01_2.mat'
        mat_data = sio.loadmat(mat_filename)
        co2 = mat_data['Color_Para']

bfm = FFTfMorphableModel(param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz', model_path='/opt/data/BFM/BFM.mat')

image1 = render(
    pose_param=tf.constant(pose1),
    shape_param=tf.constant(shape1),
    exp_param=tf.constant(exp1),
    tex_param=tf.constant(tex1),
    color_param=tf.constant(co1),
    illum_param=tf.constant(il1),
    frame_height=256,
    frame_width=256,
    tf_bfm=bfm
)

image2 = render(
    pose_param=tf.constant(pose2),
    shape_param=tf.constant(shape2),
    exp_param=tf.constant(exp2),
    tex_param=tf.constant(tex2),
    color_param=tf.constant(co2),
    illum_param=tf.constant(il2),
    frame_height=256,
    frame_width=256,
    tf_bfm=bfm
)

save_images(
    images=[image1.numpy().astype(np.uint8), image2.numpy().astype(np.uint8)],
    landmarks=None,
    titles=['orignal1', 'original2'],
    filename='./test_image.jpg'
)