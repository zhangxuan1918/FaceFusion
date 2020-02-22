import tensorflow as tf

from create_tfrecord.export_tfrecord_util import split_300W_LP_labels
from training import dataset


def display(tfrecord_dir):
    print('Loading dataset %s' % tfrecord_dir)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    dset = dataset.TFRecordDataset(
        tfrecord_dir, batch_size=1, max_label_size='full', repeat=False, shuffle_mb=0)
    import cv2

    idx = 0
    while True:
        try:
            image_tensor, labels_tensor = dset.get_minibatch_tf()
            image = image_tensor[0].numpy()
        except tf.errors.OutOfRangeError:
            break

        # TODO: render images using labels
        # roi, lm, pp, shape_para, exp_para, color_para, illum_para, tex_para = split_300W_LP_labels(labels_tensor)
        if idx == 0:
            print('Displaying images')
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d' % idx)
        cv2.imshow('dataset_tool', image[:, :, ::-1])
        idx += 1
        if cv2.waitKey() == 27:
            break

    print('\nDisplayed %d images' % idx)


if __name__ == '__main__':
    tfrecord_dir = '/opt/data/face-fuse/test/'
    display(tfrecord_dir)