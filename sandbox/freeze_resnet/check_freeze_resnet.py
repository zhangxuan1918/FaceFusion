from project_code.models.old.networks_linear_3dmm import Face3DMM
import tensorflow as tf

checkpoint_dir = 'G:/PycharmProjects/FaceFusion/project_code/data/pretrained_model/20190608/'

face_model = Face3DMM()
face_model.build(input_shape=(None, 224, 224, 3))
ckpt = tf.train.Checkpoint(step=tf.Variable(0), net=face_model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

face_model.freeze_resnet()
print(face_model.summary())