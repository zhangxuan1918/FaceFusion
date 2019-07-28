import tensorflow as tf

from data_tools.data_const import face_vgg2_input_mean


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

        if param == 'landmark':
            # we didn't rescale landmarks, thus we have to resale the loss
            param_loss /= 100.
        G_loss += param_loss * loss_weights[param]

    return G_loss


def loss_3dmm(face_vgg2, images, images_rendered, metric, loss_type):
    # original images
    gt = face_vgg2(images, training=False)

    # rendered images
    images_rendered -= face_vgg2_input_mean
    est = face_vgg2(images_rendered, training=False)

    # compute loss
    G_loss = loss_norm(est=est, gt=gt, loss_type=loss_type) + 0.3 * loss_norm(est=images_rendered, gt=images, loss_type=loss_type)

    metric.update_state(G_loss)
    return G_loss
