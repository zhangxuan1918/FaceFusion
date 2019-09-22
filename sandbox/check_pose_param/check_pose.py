import numpy as np

from morphable_model.model.morphable_model import FFTfMorphableModel
import scipy.io as sio


bfm = FFTfMorphableModel(param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz', model_path='/opt/data/BFM/BFM.mat')

pic_name = 'IBUG_image_008_1_0'
# pic_name = 'IBUG_image_014_01_2'
mat_filename = '../../examples/Data/{0}.mat'.format(pic_name)
mat_data = sio.loadmat(mat_filename)

sp = mat_data['Shape_Para']
ep = mat_data['Exp_Para']
tp = mat_data['Tex_Para'][:40, :]
cp = mat_data['Color_Para']
ip = mat_data['Illum_Para']
pp = mat_data['Pose_Para']

# normalize data
sp = np.divide(np.subtract(sp, bfm.stats_shape_mu.numpy()), bfm.stats_shape_std.numpy())
ep = np.divide(np.subtract(ep, bfm.stats_exp_mu.numpy()), bfm.stats_exp_std.numpy())
tp = np.divide(np.subtract(tp, bfm.stats_tex_mu.numpy()), bfm.stats_tex_std.numpy())
cp = np.divide(np.subtract(cp, bfm.stats_color_mu.numpy()), bfm.stats_color_std.numpy())
ip = np.divide(np.subtract(ip, bfm.stats_illum_mu.numpy()), bfm.stats_illum_std.numpy())

# update pose param with random translation
pp[0, 3:5] = 256 / 450.
pp[0, 6] = 256 / 450.
pp[0, 3], pp[0, 4] = pp[0, 3] - 12, pp[0, 4] - (32. - 24)
pp = np.divide(np.subtract(pp, bfm.stats_pose_mu.numpy()), bfm.stats_pose_std.numpy())

print(pp)