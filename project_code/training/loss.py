import tensorflow as tf

from data_tools.data_const import face_vgg2_input_mean


def loss_norm(est, gt, loss_type, param):
    """
    compute l1 or l2 loss
    :param est: estimations
    :param gt: true labels
    :param loss_type: l1 or l2
    :param param: name of the parameter
    :return:
    """
    # treat pose param differently
    if param == 'pose':
        # add more weights to angle, translation
        diff = tf.square(est - gt)
        return 100 * tf.reduce_mean(diff[:, 0, 0:3]) + tf.reduce_mean(diff[:, 0, 3:])
    else:
        if loss_type == 'l2':
            return tf.reduce_mean(tf.square(est - gt))
        elif loss_type == 'l1':
            return tf.reduce_mean(tf.abs(est - gt))
        else:
            raise Exception('unsupported loss_type={0}'.format(loss_type))


def loss_3dmm_warmup(gt: dict, est: dict, metric: dict, loss_types: dict, loss_weights: dict, is_use_loss_landmark: bool):

    G_loss = 0
    loss_info = ''
    for param in gt:

        if not is_use_loss_landmark and param == 'landmark':
            continue

        param_loss = loss_norm(
            est=est[param],
            gt=gt[param],
            loss_type=loss_types[param],
            param=param
        )

        if param in metric:
            metric[param].update_state(param_loss)

        if param == 'landmark':
            # we didn't rescale landmarks, thus we have to resale the loss
            param_loss /= 450.
        G_loss += param_loss * loss_weights[param]

        loss_info += '%s: %.3f(%d); ' % (param, param_loss.numpy(), loss_weights[param])
    loss_info = ('total loss: %.3f; ' % G_loss.numpy()) + loss_info
    return G_loss, loss_info


def loss_3dmm(face_vgg2, images, images_rendered, metric, loss_type):
    # original images
    gt = face_vgg2(images, training=False)

    # rendered images
    images_rendered -= face_vgg2_input_mean
    est = face_vgg2(images_rendered, training=False)

    # compute loss
    G_loss_image = loss_norm(est=images_rendered, gt=images, loss_type=loss_type)
    G_loss_face_vgg2 = loss_norm(est=est, gt=gt, loss_type=loss_type)
    G_loss = G_loss_face_vgg2 + 0.3 * G_loss_image
    loss_info = 'total loss: %.3f; image loss: %.3f; feature loss: %.3f' % (G_loss.numpy(), G_loss_image.numpy(), G_loss_face_vgg2.numpy())
    metric.update_state(G_loss)
    return G_loss, loss_info
