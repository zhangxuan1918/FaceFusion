import tensorflow as tf
from tf_3dmm.mesh.reader import render_batch
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from create_tfrecord.export_tfrecord_util import split_300W_LP_labels, fn_unnormalize_300W_LP_labels
from training import dataset
import numpy as np
import imageio


def display(tfrecord_dir, bfm_path, image_size, num_images=5, n_tex_para=40):
    print('Loading sdataset %s' % tfrecord_dir)

    batch_size = 4
    dset = dataset.TFRecordDataset(
        tfrecord_dir, batch_size=batch_size, max_label_size='full', repeat=False, shuffle_mb=0)

    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        n_tex_para=n_tex_para
    )

    idx = 0
    filename = '/opt/project/output/verify_dataset/20200222/image_batch_{0}_indx_{1}.jpg'
    fn_unnormalize_labels = fn_unnormalize_300W_LP_labels(bfm_path=bfm_path, image_size=image_size, n_tex_para=n_tex_para)
    while idx < num_images:
        try:
            image_tensor, labels_tensor = dset.get_minibatch_tf()
        except tf.errors.OutOfRangeError:
            break

        if idx == 4:
            t = 1

        # render images using labels
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = split_300W_LP_labels(labels_tensor)
        roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para = fn_unnormalize_labels(roi, landmarks, pose_para, shape_para, exp_para, color_para, illum_para, tex_para)
        image_rendered = render_batch(
            pose_param=pose_para,
            shape_param=shape_para,
            exp_param=exp_para,
            tex_param=tex_para,
            color_param=color_para,
            illum_param=illum_para,
            frame_height=image_size,
            frame_width=image_size,
            tf_bfm=bfm,
            batch_size=batch_size
        ).numpy().astype(np.uint8)

        for i in range(batch_size):

            # images = np.concatenate((image_tensor[i].numpy().astype(np.uint8), image_rendered[i]), axis=0)
            images = image_rendered[i]
            imageio.imsave(filename.format(idx, i), images)
        idx += 1

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':
    tfrecord_dir = '/opt/data/face-fuse/test/'
    bfm_path = '/opt/data/BFM/BFM.mat'
    image_size = 450
    num_images = 8
    display(tfrecord_dir, bfm_path, image_size, num_images=num_images)