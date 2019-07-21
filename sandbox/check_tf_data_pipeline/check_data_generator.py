from data_tools.data_generator import get_3dmm_warmup_data
from morphable_model.model.morphable_model import FFTfMorphableModel

bfm = FFTfMorphableModel(
    param_mean_var_path='/opt/data/300W_LP_stats/stats_300W_LP.npz',
    model_path='/opt/data/BFM/BFM.mat',
    model_type='BFM'
)
train_ds, test_ds = get_3dmm_warmup_data(
    bfm=bfm,
    data_train_dir='/opt/data/300W_LP/',
    data_test_dir='/opt/data/AFLW2000/'
)

for epoch in range(1):
    for batch_id, values in enumerate(train_ds):
        print(batch_id)
