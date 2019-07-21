import tensorflow as tf


def loss_norm(est, gt, loss_type):
    """
    compute l1 or l2 loss
    :param est: estimations
    :param gt: true labels
    :param loss_type: l1 or l2
    :return:
    """
    if loss_type == 'l2':
        return tf.reduce_mean(tf.square(est - gt))
    elif loss_type == 'l1':
        return tf.reduce_mean(tf.abs(est - gt))
    else:
        raise Exception('unsupported loss_type={0}'.format(loss_type))


def loss_3dmm_warmup(gt: dict, est: dict, metric: dict, loss_types: dict, loss_weights: dict):

    G_loss = 0
    for param in gt:

        param_loss = loss_norm(
            est=est[param],
            gt=gt[param],
            loss_type=loss_types[param]
        )
        metric[param].update_state(param_loss)
        G_loss += param_loss * loss_weights[param]

    return G_loss


def loss_3dmm(images, images_rendered, metric, loss_type):
    images_masked = tf.where(tf.greater(images_rendered, 0), images, images_rendered)
    G_loss = loss_norm(est=images_rendered, gt=images_masked, loss_type=loss_type)

    metric.update_state(G_loss)
    return G_loss
