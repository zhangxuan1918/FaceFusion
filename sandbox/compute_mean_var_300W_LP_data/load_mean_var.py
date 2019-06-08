import numpy as np

params_mean_var = np.load('G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\\300W_LP_mean_var\stats_300W_LP.npz')

for key in params_mean_var:
    print(key)