import tensorflow as tf


def loss_norm(est, label, loss_type):
    """
    compute l1 or l2 loss
    :param est: estimations
    :param label: true labels
    :param loss_type: l1 or l2
    :return:
    """
    if loss_type == 'l2':
        return tf.reduce_mean(tf.square(est - label))
    elif loss_type == 'l1':
        return tf.reduce_mean(tf.abs(est - label))
    else:
        raise Exception('unsupported loss_type={0}'.format(loss_type))