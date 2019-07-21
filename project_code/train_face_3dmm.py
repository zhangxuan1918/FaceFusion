from models.networks_linear_3dmm import FaceNetLinear3DMM
from training.config_util import EasyDict


def config_general_settings() -> EasyDict:
    config = EasyDict()
    config['is_using_warmup'] = True
    config['dim_illum'] = 10
    config['dim_color'] = 7
    config['dim_tex'] = 199
    config['dim_shape'] = 199
    config['dim_exp'] = 29
    config['dim_pose'] = 7
    config['save_dir'] = '/opt/project/project_code/data/face_3dmm_models/20190720/'
    config['bfm_dir'] = '/opt/data/BFM/BFM.mat'

    return config


def config_train_warmup_settings() -> EasyDict:

    config = EasyDict()
    config['loss_shape_type'] = 'l2'
    config['loss_pose_type'] = 'l2'
    config['loss_exp_type'] = 'l2'
    config['loss_color_type'] = 'l2'
    config['loss_tex_type'] = 'l2'
    config['loss_landmark_type'] = 'l2'
    config['loss_illum_type'] = 'l2'

    config['face_vgg_v2_path'] = '/opt/data/face_vgg_v2/weights.h5'
    config['log_freq'] = 100
    config['eval_freq'] = 100

    config['max_checkpoint_to_keep'] = 5
    config['data_train_dir'] = '/opt/data/300W_LP/'
    config['data_test_dir'] = '/opt/data/AFLW2000/'
    config['data_mean_std'] = '/opt/data/300W_LP_stats/stats_300W_LP.npz'

    config['batch_size'] = 4
    config['learning_rate'] = 0.001
    config['beta_1'] = 0.9

    return config


def config_train_settings() -> EasyDict:

    config = EasyDict()
    config['loss_type'] = 'l2'
    config['log_freq'] = 100
    config['max_checkpoint_to_keep'] = 5

    config['batch_size'] = 4

    return config


def train_face_3dmm():
    config_general = config_general_settings()
    config_train_warmup = config_train_warmup_settings()
    config_train = config_train_settings()

    model = FaceNetLinear3DMM(
        config_general=config_general,
        config_train=config_train,
        config_train_warmup=config_train_warmup
    )

    model.train(numof_epochs_warmup=5, numof_epochs=0)


if __name__ == '__main__':
    train_face_3dmm()