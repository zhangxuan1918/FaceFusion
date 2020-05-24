from project_code.create_tfrecord.export_tfrecord_util import fn_extract_300W_LP_labels, split_300W_LP_labels, \
    fn_unnormalize_300W_LP_labels
import scipy.io as sio
import tensorflow as tf

param_mean_std_path = '/opt/data/face-fuse/stats_300W_LP.npz'

normlize_fn = fn_extract_300W_LP_labels(param_mean_std_path, 450, is_aflw_2000=True)
unnormalize_labels = fn_unnormalize_300W_LP_labels(param_mean_std_path=param_mean_std_path, image_size=450)

filename = 'image00002'
img_filename = '/opt/data/AFLW2000/' + filename + '.jpg'
mat_filename = '/opt/data/AFLW2000/' + filename + '.mat'
data_raw = sio.loadmat(mat_filename)
data = normlize_fn(img_filename=img_filename)

tf_data = tf.constant(data, dtype=tf.float32)
tf_data = tf.reshape(tf_data, (1, 430))
roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = split_300W_LP_labels(tf_data)
roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = unnormalize_labels(roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para)


t = 1