import tensorflow as tf


def display(tfrecord_dir):
    print('Loading dataset %s' % tfrecord_dir)
    tf.config.gpu.set_per_process_memory_growth(True)
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    import cv2

    idx = 0
    while True:
        try:
            images, labels = dset.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            break

        # TODO: render images using labels

        if idx == 0:
            print('Displaying images')
            cv2.namedWindow('dataset_tool')
            print('Press SPACE or ENTER to advance, ESC to exit')
        print('\nidx = %-8d' % idx)
        cv2.imshow('dataset_tool', images[0][:, :, ::-1])
        idx += 1
        if cv2.waitKey() == 27:
            break

    print('\nDisplayed %d images' % idx)