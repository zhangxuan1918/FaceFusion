import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import numpy as np

data_folder = '/opt/data/300W_LP/'
data_root = pathlib.Path(data_folder)
mat_paths = list(data_root.glob('*/*.mat'))
param_mean_std = np.load('/opt/data/face-fuse/stats_300W_LP.npz')


def process_data(data_name):
    data = []
    counter = 0

    for mat_file in mat_paths:
        mat_contents = sio.loadmat(str(mat_file))
        counter += 1

        param = np.copy(mat_contents[data_name])

        if data_name == 'Pose_Para':
            param[0, 3:5] = param[0, 3:5] / 450.
        elif data_name == 'Tex_Para':
            param = param[:40, :]
        elif data_name == 'Color_Para':
            param = param[:, :6]
        elif data_name == 'Illum_Para':
            param = param[:, :9]
        param -= param_mean_std[data_name + '_mean']
        param /= param_mean_std[data_name + '_std']
        # print(pose_para)
        data.append(param)

        if counter % 1000 == 0:
            print('counter= %d' % counter)
            # print(pose_para_mean)
            # print(pose_para_var / (counter - 1))
            #
            # print(color_para_var)
            # print(color_para_var / (counter - 1))
    data = np.array(data)
    if data_name in ['Shape_Para', 'Exp_Para', 'Tex_Para']:
        data = np.transpose(data, (0, 2, 1))
    return np.array(data)


def plot_his(data, filename, n_bins=100):
    fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)

    axs[0, 0].hist(data[:, 0, 0], bins=n_bins)
    axs[0, 1].hist(data[:, 0, 1], bins=n_bins)
    axs[0, 2].hist(data[:, 0, 2], bins=n_bins)
    # translation
    axs[1, 0].hist(data[:, 0, 3], bins=n_bins)
    axs[1, 1].hist(data[:, 0, 4], bins=n_bins)
    axs[1, 2].hist(data[:, 0, 5], bins=n_bins)

    # scaling
    # axs[2, 0].hist(data[:, 0, 6], bins=n_bins)

    fig.savefig(filename)

data_name = 'Illum_Para'
data = process_data(data_name)
plot_his(data[:, :, :7], './%s.png' % data_name, n_bins='auto')
