from models.networks_linear_3dmm import FaceNetLinear3DMM
from training.config_util import EasyDict


def config_general_settings(input_image_size) -> EasyDict:
    config = EasyDict()
    config['is_using_warmup'] = True
    config['dim_illum'] = 10
    config['dim_color'] = 7
    config['dim_tex'] = 40
    config['dim_shape'] = 199
    config['dim_exp'] = 29
    config['dim_pose'] = 7

    config['dim_gf'] = 32
    config['dim_gfc'] = 512

    config['save_dir'] = '/opt/project/project_code/data/face_3dmm_models/20190915/'
    config['bfm_dir'] = '/opt/data/BFM/BFM.mat'
    config['input_image_size'] = input_image_size
    return config


def config_train_warmup_settings(input_image_size) -> EasyDict:

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
    config['eval_freq'] = 1000
    config['input_image_size'] = input_image_size

    config['max_checkpoint_to_keep'] = 5
    config['data_train_dir'] = '/opt/data/300W_LP/'
    config['data_test_dir'] = '/opt/data/AFLW2000/'
    config['data_mean_std'] = '/opt/data/300W_LP_stats/stats_300W_LP.npz'

    config['num_of_epochs'] = 100
    config['batch_size'] = 64
    config['learning_rate'] = 0.0002
    config['beta_1'] = 0.5

    return config


def config_train_settings(input_image_size) -> EasyDict:

    config = EasyDict()
    config['loss_type'] = 'l2'
    config['max_checkpoint_to_keep'] = 5
    config['log_freq'] = 100
    config['eval_freq'] = 100

    config['input_image_size'] = input_image_size

    config['num_of_epochs'] = 10
    config['batch_size'] = 64
    config['learning_rate'] = 0.0002
    config['beta_1'] = 0.5

    config['data_train_dir'] = '/opt/data/ffhq-dataset/'
    config['data_test_dir'] = '/opt/data/ffhq-dataset-test'

    return config


def train_face_3dmm():
    config_general = config_general_settings(input_image_size=224)
    config_train_warmup = config_train_warmup_settings(input_image_size=224)
    config_train = config_train_settings(input_image_size=224)

    model = FaceNetLinear3DMM(
        config_general=config_general,
        config_train=config_train,
        config_train_warmup=config_train_warmup
    )

    model.train()


if __name__ == '__main__':
    train_face_3dmm()