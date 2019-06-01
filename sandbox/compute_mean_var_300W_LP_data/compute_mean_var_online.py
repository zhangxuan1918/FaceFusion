import pathlib

import numpy as np
import scipy.io as sio

data_folder = 'H:/300W-LP/300W_LP'
data_root = pathlib.Path(data_folder)
mat_paths = list(data_root.glob('*/*.mat'))

shape_para_mean = np.zeros((199, 1))
shape_para_var = np.zeros((199, 1))
pose_para_mean = np.zeros((1, 7))
pose_para_var = np.zeros((1, 7))
exp_para_mean = np.zeros((29, 1))
exp_para_var = np.zeros((29, 1))
color_para_mean = np.zeros((1, 7))
color_para_var = np.zeros((1, 7))
illum_para_mean = np.zeros((1, 10))
illum_para_var = np.zeros((1, 10))
pt2d_mean = np.zeros((136, 1))
pt2d_var = np.zeros((136, 1))
tex_para_mean = np.zeros((199, 1))
tex_para_var = np.zeros((199, 1))

counter = 0


def compute_mean_var(x, x_mean, x_var, n):
    delta = x - x_mean
    x_mean = x_mean + delta / n
    x_var = x_var + delta * (x - x_mean)
    return x_mean, x_var


keys = ['roi', 'Shape_Para', 'Pose_Para', 'Exp_Para', 'Color_Para', 'Illum_Para', 'pt2d', 'Tex_Para']
for mat_file in mat_paths:
    mat_contents = sio.loadmat(str(mat_file))
    counter += 1

    # roi_mean, roi_var = compute_mean_var(mat_contents['roi'], roi_mean, roi_var, counter)
    shape_para_mean, shape_para_var = compute_mean_var(mat_contents['Shape_Para'], shape_para_mean, shape_para_var, counter)
    pose_para_mean, pose_para_var = compute_mean_var(mat_contents['Pose_Para'], pose_para_mean, pose_para_var, counter)
    exp_para_mean, exp_para_var = compute_mean_var(mat_contents['Exp_Para'], exp_para_mean, exp_para_var, counter)
    color_para_mean, color_para_var = compute_mean_var(mat_contents['Color_Para'], color_para_mean, color_para_var, counter)
    illum_para_mean, illum_para_var = compute_mean_var(mat_contents['Illum_Para'], illum_para_mean, illum_para_var, counter)
    pt2d_mean, pt2d_var = compute_mean_var(np.reshape(mat_contents['pt2d'], (-1, 1)), pt2d_mean, pt2d_var, counter)
    tex_para_mean, tex_para_var = compute_mean_var(mat_contents['Tex_Para'], tex_para_mean, tex_para_var, counter)

    if counter % 1000 == 0:
        print('counter= %d' % counter)
        print(pose_para_mean)
        print(pose_para_var / (counter - 1)
              )
np.savez('stats_300W_LP', shape_para_mean, shape_para_var / (counter - 1),
         pose_para_mean, pose_para_var / (counter - 1), exp_para_mean, exp_para_var / (counter - 1),
         color_para_mean, color_para_var / (counter - 1), illum_para_mean, illum_para_var / (counter - 1),
         pt2d_mean, pt2d_var / (counter - 1), tex_para_mean, tex_para_var / (counter - 1))



