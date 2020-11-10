import keras.backend as K
import tensorflow as tf
import numpy as np
from dataGenerator import get_anchors


def det_loss(args, n_classes, anchors, strides, input_shape):
    # *y_true: [B,H,W,a,4+c+1] for each resolution, start by x8level3
    # *cls_pred: [B,H,W,a,c] for each resolution
    # *box_pred: [B,H,W,a,4] for each resolution
    n_levels = len(strides)
    anchors = np.float32(get_anchors(anchors, strides))

    cls_preds = args[:n_levels]
    box_preds = args[n_levels:n_levels*2]
    y_true = args[n_levels*2:]

    # tranverse strides
    loss = 0.
    for i, s in enumerate(strides):
        labels = y_true[i][...,-1:]
        # cls: on positives & negatives
        cls_loss = focal_loss(y_true[i][...,4:4+n_classes], cls_preds[i], labels, from_logits=True)
        # reg: on positives
        box_true = origin2offset(y_true[i][...,:4], anchors[i], input_shape)
        box_loss = smooth_l1(box_true, box_preds[i], labels, from_logits=True)

        loss += cls_loss + box_loss

        loss = tf.Print(loss, [cls_loss, box_loss], message="  focal loss & smooth l1 loss: ")

    return loss


def focal_loss(cls_true, cls_pred, labels, alpha=0.25, gamma=2.0, from_logits=True):
    if from_logits:
        cls_pred = K.sigmoid(cls_pred)
    alpha_pos = tf.where(labels>0., labels*alpha, tf.zeros_like(labels))
    alpha_neg = tf.where(labels==0., (1-labels)*(1-alpha), tf.zeros_like(labels))
    alpha_fac = alpha_pos + alpha_neg
    pt = 1 - K.abs(cls_pred-cls_true)
    pt = K.clip(pt, K.epsilon(), 1-K.epsilon())
    focal_loss_ = -K.pow(1-pt, gamma) * K.log(pt) * alpha_fac
    norm_term = K.maximum(1., K.sum(alpha_pos, axis=[1,2,3,4]))
    focal_loss_ = K.sum(focal_loss_, axis=[1,2,3,4]) / norm_term
    return K.mean(focal_loss_)


def smooth_l1(box_true, box_pred, labels, sigma=.2, from_logits=True):
    # |x|<1: 0.5*(sigma*x)*(sigma*x), |x|>1/sigma/sigma: |x|-0.5/sigma/sigma
    # if from_logits:
    #     box_pred_txy = K.tanh(box_pred[...,:2])
    #     box_pred_twh = K.relu(box_pred[...,2:])
    #     box_pred = K.concatenate([box_pred_txy, box_pred_twh])
    valid_mask = tf.where(labels>0, tf.ones_like(labels), tf.zeros_like(labels))
    pt = tf.abs(box_true - box_pred)
    smooth_l1_ = tf.where(pt<sigma, 0.5*pt*pt/sigma, pt-0.5/sigma)
    smooth_l1_ = K.sum(smooth_l1_*valid_mask, axis=[1,2,3,4])
    return K.mean(smooth_l1_)


def origin2offset(box_true, anchors, input_shape):
    # box_true: rela-origin-[xc,yc,w,h]
    # return: rpn offset [offset_x, offset_y, offset_w, offset_h]
    #                     offset_x = (xc-a_xc)/wa
    #                     offset_y = (yc-a_yc)/ha
    #                     offset_w = log(w/wa)
    #                     offset_h = log(h/ha)

    grid_shape = K.int_shape(box_true)[1:3]    # h,w
    grid_y = K.tile(K.reshape(K.arange(0, grid_shape[0]), [1,-1,1,1,1]), [1, 1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, grid_shape[1]), [1,1,-1,1,1]), [1, grid_shape[0], 1, 1, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), tf.float32)

    norm_acxy = (grid + 0.5)/grid_shape[::-1]
    norm_awh = K.reshape(anchors, [1,1,1,anchors.shape[0], anchors.shape[1]]) / input_shape[::-1]

    offset_xy = tf.where(box_true[...,:2]>0., (box_true[...,:2] - norm_acxy) / norm_awh,
                         tf.zeros_like(box_true[...,:2]))
    offset_wh = tf.where(box_true[...,:2]>0., K.log(K.epsilon() + box_true[...,2:] / norm_awh),
                         tf.zeros_like(box_true[...,:2]))

    return K.concatenate([offset_xy, offset_wh])










