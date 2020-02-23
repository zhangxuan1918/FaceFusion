import tensorflow as tf
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel
from create_tfrecord.export_tfrecord_util import split_300W_LP_labels, fn_unnormalize_300W_LP_labels
from training import dataset
from tf_3dmm.mesh.render import render_2
import numpy as np


def display(tfrecord_dir, bfm_path, image_size):
    print('Loading dataset %s' % tfrecord_dir)

    dset = dataset.TFRecordDataset(
        tfrecord_dir, batch_size=1, max_label_size='full', repeat=False, shuffle_mb=0)

    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        model_type='BFM'
    )

    import cv2

    idx = 0
    filename = '/opt/project/output/verify_dataset/20200222/image_{0}.jpg'
    fn_unnormalize_labels = fn_unnormalize_300W_LP_labels(bfm_path=bfm_path, image_size=image_size)
    while idx < 5:
        try:
            image_tensor, labels_tensor = dset.get_minibatch_tf()
            image = image_tensor[0].numpy()
        except tf.errors.OutOfRangeError:
            break

        # render images using labels
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = split_300W_LP_labels(labels_tensor)
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = fn_unnormalize_labels(roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para)
        image_rendered = render_2(
            angles_grad=pose_para[0, 0, 0:3],
            t3d=pose_para[0, 0, 3:6],
            scaling=pose_para[0, 0, 6],
            shape_param=shape_para[0],
            exp_param=exp_para[0],
            tex_param=tex_para[0],
            color_param=color_para[0],
            illum_param=illum_para[0],
            frame_height=224,
            frame_width=224,
            tf_bfm=bfm
        ).numpy()

        images = np.concatenate((image, image_rendered), axis=0)
        cv2.imwrite(filename.format(idx), images)
        idx += 1

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':
    tfrecord_dir = '/opt/data/face-fuse/test/'
    bfm_path = '/opt/data/BFM/BFM.mat'
    image_size = 224

    display(tfrecord_dir, bfm_path, image_size)