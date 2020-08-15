import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from project_code.training import dataset


def display(tfrecord_dir, image_size, is_augment, num_images=5):
    print('Loading sdataset %s' % tfrecord_dir)

    batch_size = 4
    dset = dataset.TFRecordDatasetSupervised(
        tfrecord_dir=tfrecord_dir, batch_size=batch_size, repeat=False, shuffle_mb=0, is_augment=is_augment)

    idx = 0
    filename = '/opt/project/output/verify_dataset/supervised-ffhq-arg/20200818/image_batch_{0}_indx_{1}.jpg'

    while idx < num_images:
        try:
            image_tensor, lms_tensor = dset.get_minibatch_tf()
        except tf.errors.OutOfRangeError:
            break

        landmarks = tf.reshape(lms_tensor, shape=(-1, 2, 68)).numpy() * image_size

        for i in range(batch_size):
            print(tf.shape(image_tensor[i]))
            fig = plt.figure()
            # input image
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image_tensor[i].numpy().astype(np.uint8))
            ax.plot(landmarks[i, 0, 0:17], landmarks[i, 1, 0:17], marker='o', markersize=2, linestyle='-',
                    color='w', lw=2)
            ax.plot(landmarks[i, 0, 17:22], landmarks[i, 1, 17:22], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 22:27], landmarks[i, 1, 22:27], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 27:31], landmarks[i, 1, 27:31], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 31:36], landmarks[i, 1, 31:36], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 36:42], landmarks[i, 1, 36:42], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 42:48], landmarks[i, 1, 42:48], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 48:60], landmarks[i, 1, 48:60], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)
            ax.plot(landmarks[i, 0, 60:68], landmarks[i, 1, 60:68], marker='o', markersize=2,
                    linestyle='-', color='w', lw=2)

            plt.savefig(filename.format(idx, i))

        idx += 1


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
    # training data is without augment
    tfrecord_dir = '/opt/data/face-fuse/supervised_ffhq_arg/train/'
    is_augment = True

    # testing data is without augment
    # tfrecord_dir = '/opt/data/face-fuse/supervised_ffhq_arg/test/'
    # is_augment = False


    image_size = 224
    num_images = 16
    display(tfrecord_dir, image_size, is_augment, num_images=num_images)