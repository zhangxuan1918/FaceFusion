import os
import numpy as np

shape_para_mean = np.zeros(100)
shape_para_var = np.zeros(100)
pose_para_mean = np.zeros(6)
pose_para_var = np.zeros(6)
exp_para_mean = np.zeros(79)
exp_para_var = np.zeros(79)

stats_300lp = np.load('/opt/data/face-fuse/stats_300W_LP.npz')
color_para_mean = np.squeeze(stats_300lp['Color_Para_mean'])
color_para_std = np.squeeze(stats_300lp['Color_Para_std'])
tex_para_mean = np.squeeze(stats_300lp['Tex_Para_mean'])
tex_para_std = np.squeeze(stats_300lp['Tex_Para_std'])
illum_para_mean = np.squeeze(stats_300lp['Illum_Para_mean'])
illum_para_std = np.squeeze(stats_300lp['Illum_Para_std'])



def compute_mean_var(x, x_mean, x_var, n):
    delta = x - x_mean
    x_mean = x_mean + delta / n
    x_var = x_var + delta * (x - x_mean)
    return x_mean, x_var


f_80k_folder = '/opt/data/face-80k/Coarse_Dataset/CoarseData'
counter = 0

# list all subfolders
for sf in os.listdir(f_80k_folder):
    # list all files
    subdir = os.path.join(f_80k_folder, sf)
    for file in os.listdir(subdir):
        if file.endswith('.txt'):
            with open(os.path.join(subdir, file), 'r') as f:
                # read shape parameters
                row = f.readline()
                param_shape = np.asarray([float(v.strip()) for v in row.split()])
                # read expression parameters
                row = f.readline()
                param_exp = np.asarray([float(v.strip()) for v in row.split()])
                # read pose parameters
                row = f.readline()
                param_pose = np.asarray([float(v.strip()) for v in row.split()])
                param_pose[3:] = param_pose[3:] / 450.
                counter += 1

                shape_para_mean, shape_para_var = compute_mean_var(param_shape, shape_para_mean, shape_para_var,
                                                                   counter)
                exp_para_mean, exp_para_var = compute_mean_var(param_exp, exp_para_mean, exp_para_var, counter)
                pose_para_mean, pose_para_var = compute_mean_var(param_pose, pose_para_mean, pose_para_var, counter)

                if counter % 1000 == 0:
                    print('counter= %d' % counter)

np.savez(
    os.path.join('/opt/data/face-80k/Coarse_Dataset/', 'stats_80k'),
    Shape_Para_mean=shape_para_mean,
    Shape_Para_std=np.sqrt(shape_para_var / (counter - 1)),
    Pose_Para_mean=pose_para_mean,
    Pose_Para_std=np.sqrt(pose_para_var / (counter - 1)),
    Exp_Para_mean=exp_para_mean,
    Exp_Para_std=np.sqrt(exp_para_var / (counter - 1)),
    Color_Para_mean=color_para_mean,
    Color_Para_std=color_para_std,
    Illum_Para_mean=illum_para_mean,
    Illum_Para_std=illum_para_std,
    Tex_Para_mean=tex_para_mean,
    Tex_Para_std=tex_para_std
)

print('===========pose mean============')
print(pose_para_mean)
print('===========pose std============')
print(np.sqrt(pose_para_var / (counter - 1)))
