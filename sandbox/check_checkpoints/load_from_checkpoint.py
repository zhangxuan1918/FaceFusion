import tensorflow as tf

from project_code.models.networks_linear_3dmm import Face3DMM

checkpoint_dir = 'G:\PycharmProjects\FaceFusion\project_code\data\supervised_fine_tuned_model\\2019-06-08\models'

# checkpoint_dir = 'G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190530/'

face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=face_model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

print(face_model.summary())