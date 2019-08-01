import tensorflow as tf

from models.networks_linear_3dmm import FaceNetLinear3DMM
from train_face_3dmm import config_general_settings, config_train_warmup_settings, config_train_settings
from training.data import setup_3dmm_warmup_data
from training.loss import loss_3dmm_warmup
from training.opt import compute_landmarks, save_rendered_images_for_warmup_eval

config_general = config_general_settings()
config_train_warmup = config_train_warmup_settings()
config_train = config_train_settings()
loss_types = {
        'shape': 'l2',
        'pose': 'l2',
        'exp': 'l2',
        'color': 'l2',
        'illum': 'l2',
        'tex': 'l2',
        'landmark': 'l2',
    }

loss_weights = {
    'shape': 20,
    'pose': 20,
    'exp': 20,
    'color': 5,
    'illum': 5,
    'tex': 5,
    'landmark': 10
}

face_model = FaceNetLinear3DMM(
    config_general=config_general,
    config_train=config_train,
    config_train_warmup=config_train_warmup
)

ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=face_model.model)
checkpoint_dir = '/opt/project/project_code/data/face_3dmm_models/20190727/warm_up/model/'
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
ckpt.restore(manager.latest_checkpoint)

_, test_ds = setup_3dmm_warmup_data(
    bfm=face_model.bfm,
    batch_size=8,
    data_train_dir='/opt/data/300W_LP/',
    data_test_dir='/opt/data/AFLW2000/'
)

for i, value in enumerate(test_ds):
    images, ground_truth = value

    est = face_model(images, training=False)

    est['landmark'] = compute_landmarks(
        poses_param=est.get('pose') * face_model.bfm.stats_pose_std + face_model.bfm.stats_pose_mu,
        shapes_param=est.get('shape') * face_model.bfm.stats_shape_std + face_model.bfm.stats_shape_mu,
        exps_param=est.get('exp') * face_model.bfm.stats_exp_std + face_model.bfm.stats_exp_mu,
        bfm=face_model.bfm
    )
    one_loss, loss_info = loss_3dmm_warmup(
        gt=ground_truth,
        est=est,
        metric={},
        loss_types=loss_types,
        loss_weights=loss_weights,
        is_use_loss_landmark=True
    )

    print(loss_info)

    save_rendered_images_for_warmup_eval(
        bfm=face_model.bfm,
        images=images,
        gt=ground_truth,
        est=est,
        image_size=224,
        eval_dir='./eval/',
        batch_id=i,
        num_images_to_render=8,
        max_images_in_dir=2000
    )
