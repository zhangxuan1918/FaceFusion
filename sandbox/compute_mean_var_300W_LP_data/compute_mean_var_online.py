import pathlib

import numpy as np
import scipy.io as sio

data_folder = '/opt/data/300W_LP/'
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
tex_para_mean = np.zeros((199, 1))
tex_para_var = np.zeros((199, 1))

counter = 0


def compute_mean_var(x, x_mean, x_var, n):
    delta = x - x_mean
    x_mean = x_mean + delta / n
    x_var = x_var + delta * (x - x_mean)
    return x_mean, x_var


keys = ['roi', 'Shape_Para', 'Pose_Para', 'Exp_Para', 'Color_Para', 'Illum_Para', 'pt2d', 'Tex_Para']
pose_abs_max = np.array([-10000000] * 7, dtype=np.float)
pose_abs_max = np.reshape(pose_abs_max, (1, 7))

for mat_file in mat_paths:
    mat_contents = sio.loadmat(str(mat_file))
    counter += 1

    # roi_mean, roi_var = compute_mean_var(mat_contents['roi'], roi_mean, roi_var, counter)
    shape_para_mean, shape_para_var = compute_mean_var(mat_contents['Shape_Para'], shape_para_mean, shape_para_var, counter)
    pose_para_mean, pose_para_var = compute_mean_var(mat_contents['Pose_Para'], pose_para_mean, pose_para_var, counter)
    exp_para_mean, exp_para_var = compute_mean_var(mat_contents['Exp_Para'], exp_para_mean, exp_para_var, counter)
    color_para_mean, color_para_var = compute_mean_var(mat_contents['Color_Para'], color_para_mean, color_para_var, counter)
    illum_para_mean, illum_para_var = compute_mean_var(mat_contents['Illum_Para'], illum_para_mean, illum_para_var, counter)
    tex_para_mean, tex_para_var = compute_mean_var(mat_contents['Tex_Para'], tex_para_mean, tex_para_var, counter)

    pose_abs_max = np.maximum(pose_abs_max, np.abs(mat_contents['Pose_Para']))

    if counter % 1000 == 0:
        print('counter= %d' % counter)
        # print(pose_para_mean)
        # print(pose_para_var / (counter - 1))
        #
        # print(color_para_var)
        # print(color_para_var / (counter - 1))

# replace 0 var to be 1
shape_para_var[shape_para_var < 0.000001] = 1.
pose_para_var[pose_para_var < 0.000001] = 1.
exp_para_var[exp_para_var < 0.000001] = 1.
color_para_var[color_para_var < 0.000001] = 1.
illum_para_var[illum_para_var < 0.000001] = 1.
tex_para_var[tex_para_var < 0.000001] = 1.

print('min shape var %5f' % np.min(shape_para_var))
print('min pose var %5f' % np.min(pose_para_var))
print('min exp var %5f' % np.min(exp_para_var))
print('min color var %5f' % np.min(color_para_var))
print('min illum var %5f' % np.min(illum_para_var))
print('min tex var %5f' % np.min(tex_para_var))

np.savez('stats_300W_LP',
         Shape_Para_mean=shape_para_mean,
         Shape_Para_std=np.sqrt(shape_para_var / (counter - 1)),
         Pose_Para_mean=pose_para_mean,
         Pose_Para_std=np.sqrt(pose_para_var / (counter - 1)),
         Exp_Para_mean=exp_para_mean,
         Exp_Para_std=np.sqrt(exp_para_var / (counter - 1)),
         Color_Para_mean=color_para_mean,
         Color_Para_std=np.sqrt(color_para_var / (counter - 1)),
         Illum_Para_mean=illum_para_mean,
         Illum_Para_std=np.sqrt(illum_para_var / (counter - 1)),
         Tex_Para_mean=tex_para_mean,
         Tex_Para_std=np.sqrt(tex_para_var / (counter - 1)))



