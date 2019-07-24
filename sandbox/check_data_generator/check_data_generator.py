from data_tools.data_util import load_mat_3dmm
from morphable_model.model.morphable_model import FFTfMorphableModel

bfm = FFTfMorphableModel(param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz', model_path='/opt/data/BFM/BFM.mat')

load_mat_3dmm(bfm=bfm, data_name='300W_LP', mat_file='/opt/data/300W_LP/HELEN/HELEN_2088010176_2_1.mat')
