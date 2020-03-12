import PIL
import imageio
import tensorflow as tf
from tf_3dmm.mesh.reader import render_batch
import numpy as np
from tf_3dmm.morphable_model.morphable_model import TfMorphableModel

from project_code.create_tfrecord.export_tfrecord_util import split_300W_LP_labels, unnormalize_labels
from project_code.training import dataset


def load_model(pd_model_path):
    return tf.keras.models.load_model(pd_model_path)


def check_prediction_adhoc(tfrecord_dir, bfm_path, pd_model_path, num_batches, batch_size, resolution, n_tex_para):

    dset = dataset.TFRecordDataset(tfrecord_dir, batch_size=batch_size, max_label_size='full', repeat=False, shuffle_mb=0)
    strategy = tf.distribute.MirroredStrategy()
    print('Loading BFM model')
    bfm = TfMorphableModel(
        model_path=bfm_path,
        n_tex_para=n_tex_para
    )
    model = load_model(pd_model_path=pd_model_path)

    idx = 0
    filename = '/opt/project/output/verify_dataset/20200222/image_batch_{0}_indx_{1}.jpg'

    while idx < num_batches:
        try:
            image_tensor, gt_params = dset.get_minibatch_tf()
            est_params = model(image_tensor)
        except tf.errors.OutOfRangeError:
            break

        _, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex = split_300W_LP_labels(gt_params)

        _, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex = unnormalize_labels(
            bfm, batch_size, resolution, None, gt_lm, gt_pp, gt_shape, gt_exp, gt_color, gt_illum, gt_tex)

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = split_300W_LP_labels(est_params)

        _, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex = unnormalize_labels(
            bfm, batch_size, resolution, None, est_lm, est_pp, est_shape, est_exp, est_color, est_illum, est_tex)

        gt_image = render_batch(
            pose_param=gt_pp,
            shape_param=gt_shape,
            exp_param=gt_exp,
            tex_param=gt_tex,
            color_param=gt_color,
            illum_param=gt_illum,
            frame_height=resolution,
            frame_width=resolution,
            tf_bfm=bfm,
            batch_size=batch_size
        ).numpy().astype(np.uint8)

        est_image = render_batch(
            pose_param=est_pp,
            shape_param=est_shape,
            exp_param=est_exp,
            tex_param=est_tex,
            color_param=est_color,
            illum_param=est_illum,
            frame_height=resolution,
            frame_width=resolution,
            tf_bfm=bfm,
            batch_size=batch_size
        ).numpy().astype(np.uint8)

        for i in range(batch_size):
            images = np.concatenate((gt_image[i], est_image[i]), axis=0)
            # images = image_rendered[i]
            imageio.imsave(filename.format(idx, i), images)
        idx += 1

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':
    n_tex_para = 40
    tf_bfm = TfMorphableModel(model_path='/opt/project/examples/Data/BFM/Out/BFM.mat', n_tex_para=n_tex_para)
    save_rendered_to = './output/'
    tfrecord_dir = '/opt/data/face-fuse/train/'
    bfm_path = '/opt/data/BFM/BFM.mat'
    pd_model_path = '/opt/data/face-fuse/model/20200310/supervised-exported'
    image_size = 224
    num_batches = 8
    check_prediction_adhoc(
        tfrecord_dir, bfm_path, pd_model_path, num_batches, batch_size=4, resolution=image_size, n_tex_para=n_tex_para
    )
