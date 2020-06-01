import keras.backend as K
import tensorflow as tf


def det_loss(y_true, y_pred, n_anchors, n_classes):
    # *y_true: [B,N,a*c+5*c] for each resolution
    # *y_pred: [B,N,a*c] for each resolution * [B,N,4*a] for each resolution
    n_levels = len(y_pred) // 2
    cls_preds = y_pred[:n_levels]
    box_pres = y_pred[n_levels:]
    cls_gt = y_true[:n_levels]
    box_gt = y_true[n_levels:]
    cls_losses = []
    box_losses = []
    for i in range(n_levels):
        for j in range(n_anchors):
            cls_losses.append(focal(cls_gt[i][...,j:j*n_classes], cls_preds[i][...,j:j*n_classes]))
            box_losses.append(smooth_l1(box_gt[i][...,j:j+4], box_pres[i][...,j:j+4]))
    cls_loss = tf.add_n(cls_losses)
    box_loss = tf.add_n(box_losses)
    det_loss = cls_loss + box_loss

    return det_loss


# for every resolution, for each anchor
def smooth_l1(y_true, y_pred, sigma=3.0):
    # y_true: [B,N,5], N=H*W, last channel refers to anchor state {-1:ignore, 0:negative, 1:positive}
    # y_pred: [B,N,4]
    box_pred = y_pred
    box_true = y_true[...,:-1]
    anchor_state = y_true[...,-1]

    # valid anchors
    indices = tf.where(tf.equal(anchor_state, 1))
    box_pred = tf.gather_nd(box_pred, indices)
    box_true = tf.gather_nd(box_true, indices)

    # normalizer
    normalizer = tf.cast(K.maximum(1, indices._keras_shape[1]), tf.float32)

    l1 = K.abs(box_pred-box_true)
    loss = tf.where(K.less(l1, 1./sigma**2), 0.5*(sigma*l1)**2, l1-0.5/sigma**2)
    loss = K.sum(loss) / normalizer

    return loss


# for every resolution, for each anchor
def focal(y_true, y_pred, alpha=0.25, gamma=2.0, cutoff=0.5):
    # y_true: [B,N,c+1]
    # y_pred: [B,N,c]
    cls_pred = y_pred
    cls_true = y_true[...,:-1]
    anchor_state = y_true[...,-1]

    # valid anchors
    indices = tf.where(tf.not_equal(anchor_state, -1))
    cls_pred = tf.gather_nd(cls_pred, indices)
    cls_true = tf.gather_nd(cls_true, indices)

    # normalizer
    normalizer = tf.cast(K.maximum(1, indices._keras_shape[1]), tf.float32)

    # weighting
    cls_pt = tf.where(cls_true>cutoff, cls_pred, 1-cls_pred)
    alpha_factor = tf.where(cls_true>cutoff, tf.ones_like(cls_true)*alpha, tf.ones_like(cls_true)*(1-alpha))
    focal_factor = K.pow(1-cls_pt, gamma)

    loss = -alpha_factor*focal_factor*K.log(cls_pt)
    loss = K.sum(loss) / normalizer

    return loss


