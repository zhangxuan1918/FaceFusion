import numpy as np

params_mean_var = np.load('/opt/data/300W_LP_stats/stats_300W_LP.npz')

print('pose mean')
print(params_mean_var['Pose_Para_mean'])

print('pose std')
print(params_mean_var['Pose_Para_std'])

for key in params_mean_var:
    print(key)