import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np


f_80k_folder = '/opt/data/face-80k/Coarse_Dataset/CoarseData'

data_root = pathlib.Path(f_80k_folder)
txt_paths = list(data_root.glob('*/*.txt'))
param_mean_std = np.load('/opt/data/face-fuse/stats_80k.npz')


def process_data(debug):
    shape, exp, pose = [], [], []
    counter = 0

    if debug:
        files_to_check = txt_paths[:2000]
    else:
        files_to_check = txt_paths
    for txt_file in files_to_check:
        counter += 1
        with open(os.path.join(txt_file), 'r') as f:
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

            param_shape -= param_mean_std['Shape_Para_mean']
            param_shape /= param_mean_std['Shape_Para_std']
            shape.append(param_shape)

            param_exp -= param_mean_std['Exp_Para_mean']
            param_exp /= param_mean_std['Exp_Para_std']
            exp.append(param_exp)

            param_pose -= param_mean_std['Pose_Para_mean']
            param_pose /= param_mean_std['Pose_Para_std']
            pose.append(param_pose)

            if counter % 1000 == 0:
                print('counter= %d' % counter)
    return np.asarray(shape), np.asarray(exp), np.asarray(pose)


def plot_his(data, filename, n_bins):
    fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)

    axs[0, 0].hist(data[:, 0], bins=n_bins)
    axs[0, 1].hist(data[:, 1], bins=n_bins)
    axs[0, 2].hist(data[:, 2], bins=n_bins)
    # translation
    axs[1, 0].hist(data[:, 3], bins=n_bins)
    axs[1, 1].hist(data[:, 4], bins=n_bins)
    axs[1, 2].hist(data[:, 5], bins=n_bins)

    # scaling
    # axs[2, 0].hist(data[:, 0, 6], bins=n_bins)

    fig.savefig(filename)

debug = False
shape, exp, pose = process_data(debug)
plot_his(shape[:, :6], 'shape.png', n_bins='auto')
plot_his(exp[:, :6], 'exp.png', n_bins='auto')
plot_his(pose[:, :6], 'pose.png', n_bins='auto')