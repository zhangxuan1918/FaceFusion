import imageio
import numpy as np
import tensorflow as tf

from project_code.training import dataset


def display(tfrecord_dir, num_images=5):
    print('Loading sdataset %s' % tfrecord_dir)

    batch_size = 4
    dset = dataset.TFRecordDatasetUnsupervised(
        tfrecord_dir=tfrecord_dir, batch_size=batch_size, repeat=False, shuffle_mb=0)

    idx = 0
    filename = '/opt/project/output/verify_dataset/unsupervised/20200322/image_batch_{0}_indx_{1}.jpg'
    while idx < num_images:
        try:
            image, mask = dset.get_minibatch_tf()
        except tf.errors.OutOfRangeError:
            break

        immk = tf.where(mask == 255, image, 0)
        for i in range(batch_size):
            image_mask = np.concatenate((image[i].numpy().astype(np.uint8), immk[i].numpy().astype(np.uint8)), axis=0)
            imageio.imsave(filename.format(idx, i), image_mask)
        idx += 1

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tfrecord_dir = '/opt/data/face-fuse/unsupervised/train/'
    bfm_path = '/opt/data/BFM/BFM.mat'
    image_size = 256
    num_images = 8
    display(tfrecord_dir, num_images=num_images)